"""
Mixture-of-Accent-Adapters (MoAA) runner.

Important Steps:
  1) Loads training data.
  2) Builds multi-task labels: accentedness, gender, accent.
  3) Trains Whisper-small (frozen) with:
       - pooled-state linear projection
       - multi-task heads (accentedness, accent, gender GRL)
       - soft accent codebook + neutral vector
       - adapter bank + router
       - accentedness-gated blend between adapted and original encoder states
  4) Evaluates on:
       - internal test split 
       - optional external ASR-only test sets 
  5) Saves checkpoints and CSV predictions.

Example (train + internal eval only):
  python moaa.py \
    --aesrc_train_dir /path/to/aesrc/train_shards \
    --librispeech_dir /path/to/librispeech/train-clean-100 \
    --output_root runs/moaa_exp

Example (plus external evals):
  python moaa.py \
    --aesrc_train_dir /path/to/aesrc/train_shards \
    --librispeech_dir /path/to/librispeech/train-clean-100 \
    --aesrc_test_dir /path/to/aesrc/test \
    --openslr_test_dir /path/to/openslr/test \
    --edacc_test_dir /path/to/edacc/test \
    --globe_test_dir /path/to/globe/test \
    --output_root runs/moaa_exp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import safetensors.torch as st

from datasets import Audio, DatasetDict, load_from_disk, concatenate_datasets, disable_caching
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput

disable_caching()


# =============================================================================
# Logging / misc utils
# =============================================================================

def setup_logging(log_path: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    return logging.getLogger("moaa")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu(logger: logging.Logger) -> None:
    logger.info("*" * 60)
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU available: %s | %.2f GB", torch.cuda.get_device_name(0), mem_gb)
    else:
        logger.warning("GPU not available; running on CPU.")
    logger.info("*" * 60)


def ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


# =============================================================================
# CSV saving utilities
# =============================================================================

def save_predictions_csv_multitask(
    pred_output,
    raw_ds,
    tokenizer,
    csv_path: str,
    pred_key: str = "pred_text",
    ref_key: str = "ref_text",
) -> None:
    pred_ids = pred_output.predictions
    if isinstance(pred_ids, (tuple, list)):
        pred_ids = pred_ids[0]
    pred_ids = np.asarray(pred_ids)

    label_ids = np.asarray(pred_output.label_ids).copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    ref_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    rows = []
    for i in range(len(raw_ds)):
        rows.append({
            pred_key: pred_str[i],
            ref_key: ref_str[i],
            "accented_or_not_clf": int(raw_ds[i]["accented_or_not_clf"]),
            "gender_clf": int(raw_ds[i]["gender_clf"]),
            "accent_clf": int(raw_ds[i]["accent_clf"]),
        })

    ensure_dir(Path(csv_path).parent)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")


def save_predictions_csv_external_with_clf(
    pred_output,
    trainer: Seq2SeqTrainer,
    dataset,
    raw_ds,
    tokenizer,
    csv_path: str,
    id_key: str = "utt_id",
) -> None:
    pred_ids = pred_output.predictions
    if isinstance(pred_ids, (tuple, list)):
        pred_ids = pred_ids[0]
    pred_ids = np.asarray(pred_ids)

    label_ids = np.asarray(pred_output.label_ids).copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    ref_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    model = trainer.model
    model.eval()

    pred_accbin, pred_gender, pred_accent = [], [], []
    with torch.no_grad():
        for batch in trainer.get_test_dataloader(dataset):
            input_features = batch["input_features"].to(trainer.args.device)
            clf_out = model.predict_clf(input_features)
            pred_accbin.extend(clf_out["logits_accbin"].argmax(dim=-1).cpu().tolist())
            pred_gender.extend(clf_out["logits_gender"].argmax(dim=-1).cpu().tolist())
            pred_accent.extend(clf_out["logits_accent"].argmax(dim=-1).cpu().tolist())

    assert len(raw_ds) == len(pred_text) == len(pred_accbin) == len(pred_gender) == len(pred_accent)

    rows = []
    for i in range(len(raw_ds)):
        rows.append({
            id_key: raw_ds[i].get(id_key, i),
            "pred_text": pred_text[i],
            "ref_text": ref_text[i],
            "pred_accented_or_not": int(pred_accbin[i]),
            "pred_gender": int(pred_gender[i]),
            "pred_accent": int(pred_accent[i]),
        })

    ensure_dir(Path(csv_path).parent)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")


# =============================================================================
# Dataset preparation
# =============================================================================

def normalize_sex(example: Dict[str, Any]) -> Dict[str, Any]:
    example["sex"] = str(example["sex"]).strip().lower()
    return example


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Expected columns:
      audio, text, accented_or_not_clf, gender_clf, accent_clf

    Accent label remap:
      accent_clf: 0..K  (0 == unknown)
      -> accent_labels: -100 for unknown, else accent_clf-1
    """
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids

    batch["accented_or_not_labels"] = int(batch["accented_or_not_clf"])
    batch["gender_labels"] = int(batch["gender_clf"])

    raw_accent = int(batch["accent_clf"])
    batch["accent_labels"] = -100 if raw_accent == 0 else raw_accent - 1
    return batch


def prepare_dataset_asr_only(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


# =============================================================================
# Data collators
# =============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingAndClf:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        batch["accented_or_not_labels"] = torch.tensor([f["accented_or_not_labels"] for f in features], dtype=torch.long)
        batch["gender_labels"] = torch.tensor([f["gender_labels"] for f in features], dtype=torch.long)
        batch["accent_labels"] = torch.tensor([f["accent_labels"] for f in features], dtype=torch.long)
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingASROnly:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics_asr_only(pred, tokenizer, metric_wer, metric_cer):
    predictions = pred.predictions
    gen_ids = predictions[0] if isinstance(predictions, (tuple, list)) else predictions

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "wer": 100 * metric_wer.compute(predictions=pred_str, references=label_str),
        "cer": 100 * metric_cer.compute(predictions=pred_str, references=label_str),
    }


def compute_metrics_asr_and_clf(pred, tokenizer, metric_wer, metric_cer, accent_ignore_index: int = -100):
    predictions = pred.predictions
    gen_ids = predictions[0] if isinstance(predictions, (tuple, list)) else predictions

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    if not isinstance(predictions, (tuple, list)) or len(predictions) < 10:
        return {"wer": wer, "cer": cer}

    _, logits_acc, logits_gender, logits_accent, lab_acc, lab_gender, lab_accent, lvec_acc, lvec_gender, lvec_accent = predictions

    logits_acc = np.asarray(logits_acc)
    logits_gender = np.asarray(logits_gender)
    logits_accent = np.asarray(logits_accent)

    lab_acc = np.asarray(lab_acc).astype(np.int64).reshape(-1)
    lab_gender = np.asarray(lab_gender).astype(np.int64).reshape(-1)
    lab_accent = np.asarray(lab_accent).astype(np.int64).reshape(-1)

    lvec_acc = np.asarray(lvec_acc).astype(np.float32).reshape(-1)
    lvec_gender = np.asarray(lvec_gender).astype(np.float32).reshape(-1)
    lvec_accent = np.asarray(lvec_accent).astype(np.float32).reshape(-1)

    pred_acc = logits_acc.argmax(axis=-1)
    acc_acc = float((pred_acc == lab_acc).mean())
    loss_acc = float(lvec_acc.mean())

    pred_gender = logits_gender.argmax(axis=-1)
    acc_gender = float((pred_gender == lab_gender).mean())
    loss_gender = float(lvec_gender.mean())

    mask_known = (lab_accent != int(accent_ignore_index))
    n_known = int(mask_known.sum())
    pred_accent = logits_accent.argmax(axis=-1)

    if n_known > 0:
        acc_accent = float((pred_accent[mask_known] == lab_accent[mask_known]).mean())
        loss_accent = float(lvec_accent[mask_known].mean())
    else:
        acc_accent = float("nan")
        loss_accent = float("nan")

    return {
        "wer": wer,
        "cer": cer,
        "acc_accented_or_not": acc_acc,
        "loss_accented_or_not": loss_acc,
        "acc_gender": acc_gender,
        "loss_gender": loss_gender,
        "acc_accent": acc_accent,
        "loss_accent": loss_accent,
        "n_known_accent": int(n_known),
    }


# =============================================================================
# Adapter, Router, GRL
# =============================================================================

class Adapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class AdapterRouter(nn.Module):
    def __init__(self, hidden_size: int, num_adapters: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_size, num_adapters), nn.Softmax(dim=-1))

    def forward(self, hidden_states: torch.Tensor, adapters: nn.ModuleDict):
        B, T, H = hidden_states.shape
        pooled = hidden_states.mean(dim=1)    # [B,H]
        weights = self.gate(pooled)           # [B,A]

        combined = 0
        for i, key in enumerate(adapters.keys()):
            adapted = adapters[key](hidden_states)
            w = weights[:, i].view(B, 1, 1)
            combined = combined + w * adapted
        return combined, weights


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return _GRL.apply(x, self.lambd)


# =============================================================================
# MoAA wrapper model
# =============================================================================

class WhisperAccentedConditionalAdapters(nn.Module):
    def __init__(
        self,
        whisper_model: WhisperForConditionalGeneration,
        num_adapters: int,
        bottleneck_size: int,
        num_accents: int,
        accent_ignore_index: int,
        loss_weights: Tuple[float, float, float, float],
        freeze_whisper: bool,
        grl_lambda: float,
    ):
        super().__init__()
        self.whisper = whisper_model

        H = whisper_model.model.encoder.config.d_model
        self.num_accents = int(num_accents)
        self.accent_ignore_index = int(accent_ignore_index)

        self.w_asr, self.w_accbin, self.w_accent, self.w_gender = loss_weights

        self.linear_proj = nn.Linear(H, H)

        self.head_accented = nn.Linear(H, 2)
        self.head_accent = nn.Linear(H, self.num_accents)

        self.grl_lambda = float(grl_lambda)
        self.grl = GradientReversal(lambd=self.grl_lambda)
        self.head_gender = nn.Linear(H, 2)

        self.accent_embed = nn.Embedding(self.num_accents, H)
        self.neutral_embed = nn.Parameter(torch.zeros(H))

        self.cond_proj = nn.Linear(H, H)

        self.adapters = nn.ModuleDict({f"adapter_{i}": Adapter(H, bottleneck_size) for i in range(num_adapters)})
        self.router = AdapterRouter(H, num_adapters)

        if freeze_whisper:
            for p in self.whisper.model.encoder.parameters():
                p.requires_grad = False
            for p in self.whisper.model.decoder.parameters():
                p.requires_grad = False

    @property
    def generation_config(self):
        return self.whisper.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.whisper.generation_config = value

    @property
    def config(self):
        return self.whisper.config

    @config.setter
    def config(self, value):
        self.whisper.config = value

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    def forward(
        self,
        input_features,
        decoder_input_ids=None,
        labels=None,
        accented_or_not_labels=None,
        gender_labels=None,
        accent_labels=None,
        **kwargs
    ):
        enc_out = self.whisper.model.encoder(input_features)
        hidden = enc_out.last_hidden_state
        pooled = hidden.mean(dim=1)
        pooled = self.linear_proj(pooled)

        logits_accbin = self.head_accented(pooled)
        logits_accent = self.head_accent(pooled)
        logits_gender = self.head_gender(self.grl(pooled))

        loss_accbin = loss_accent = loss_gender = None
        if accented_or_not_labels is not None:
            loss_accbin = F.cross_entropy(logits_accbin, accented_or_not_labels)
        if gender_labels is not None:
            loss_gender = F.cross_entropy(logits_gender, gender_labels)
        if accent_labels is not None:
            loss_accent = F.cross_entropy(logits_accent, accent_labels, ignore_index=self.accent_ignore_index)

        p = torch.softmax(logits_accbin, dim=-1)[:, 1]
        p_t = p.view(-1, 1, 1).to(hidden.dtype)

        accent_probs = torch.softmax(logits_accent, dim=-1)
        e_accent = accent_probs @ self.accent_embed.weight
        e_neutral = self.neutral_embed.unsqueeze(0).expand_as(e_accent)
        e = (p.unsqueeze(1) * e_accent) + ((1.0 - p).unsqueeze(1) * e_neutral)

        e_t = e.unsqueeze(1).expand(hidden.size(0), hidden.size(1), hidden.size(2))
        hidden_cond = self.cond_proj(hidden + e_t)

        adapted_all, router_w = self.router(hidden_cond, self.adapters)
        encoder_hidden = p_t * adapted_all + (1.0 - p_t) * hidden

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.shift_tokens_right(
                labels,
                self.whisper.config.pad_token_id,
                self.whisper.config.decoder_start_token_id,
            )

        dec_out = self.whisper.model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden)
        logits_asr = self.whisper.proj_out(dec_out.last_hidden_state)

        loss_asr = None
        if labels is not None:
            loss_asr = F.cross_entropy(
                logits_asr.view(-1, logits_asr.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        parts = []
        if loss_asr is not None:
            parts.append(self.w_asr * loss_asr)
        if loss_accbin is not None:
            parts.append(self.w_accbin * loss_accbin)
        if loss_accent is not None:
            parts.append(self.w_accent * loss_accent)
        if loss_gender is not None:
            parts.append(self.w_gender * loss_gender)

        total_loss = sum(parts) if parts else None

        return {
            "loss": total_loss,
            "logits": logits_asr,
            "router_weights": router_w,
            "loss_asr": loss_asr,
            "loss_accented": loss_accbin,
            "loss_accent": loss_accent,
            "loss_gender": loss_gender,
            "logits_accented_or_not": logits_accbin,
            "logits_accent": logits_accent,
            "logits_gender": logits_gender,
        }

    @torch.no_grad()
    def generate(self, input_features, attention_mask=None, **generate_kwargs):
        generate_kwargs.pop("labels", None)
        generate_kwargs.pop("accented_or_not_labels", None)
        generate_kwargs.pop("gender_labels", None)
        generate_kwargs.pop("accent_labels", None)

        if attention_mask is None:
            attention_mask = torch.ones(
                (input_features.size(0), input_features.size(-1)),
                dtype=torch.long,
                device=input_features.device,
            )

        enc_out = self.whisper.model.encoder(input_features)
        hidden = enc_out.last_hidden_state
        pooled = self.linear_proj(hidden.mean(dim=1))

        logits_accbin = self.head_accented(pooled)
        logits_accent = self.head_accent(pooled)

        p = torch.softmax(logits_accbin, dim=-1)[:, 1]
        p_t = p.view(-1, 1, 1).to(hidden.dtype)

        accent_probs = torch.softmax(logits_accent, dim=-1)
        e_accent = accent_probs @ self.accent_embed.weight
        e_neutral = self.neutral_embed.unsqueeze(0).expand_as(e_accent)
        e = (p.unsqueeze(1) * e_accent) + ((1.0 - p).unsqueeze(1) * e_neutral)

        e_t = e.unsqueeze(1).expand(hidden.size(0), hidden.size(1), hidden.size(2))
        hidden_cond = self.cond_proj(hidden + e_t)

        adapted_all, _ = self.router(hidden_cond, self.adapters)
        encoder_hidden = p_t * adapted_all + (1.0 - p_t) * hidden

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        self.whisper.generation_config.forced_decoder_ids = None

        return self.whisper.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask, **generate_kwargs)

    @torch.no_grad()
    def predict_clf(self, input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_out = self.whisper.model.encoder(input_features)
        hidden = enc_out.last_hidden_state
        pooled = self.linear_proj(hidden.mean(dim=1))

        logits_accbin = self.head_accented(pooled)
        logits_accent = self.head_accent(pooled)
        logits_gender = self.head_gender(self.grl(pooled))

        return {"logits_accbin": logits_accbin, "logits_gender": logits_gender, "logits_accent": logits_accent}

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.whisper.save_pretrained(str(save_path), safe_serialization=safe_serialization)

        flat_sd = {}

        def add(prefix: str, sd: Dict[str, torch.Tensor]):
            for k, v in sd.items():
                flat_sd[f"{prefix}.{k}"] = v.contiguous()

        add("head_accented", self.head_accented.state_dict())
        add("head_accent", self.head_accent.state_dict())
        add("head_gender", self.head_gender.state_dict())
        add("accent_embed", self.accent_embed.state_dict())
        flat_sd["neutral_embed"] = self.neutral_embed.detach().contiguous()
        add("cond_proj", self.cond_proj.state_dict())
        add("adapters", self.adapters.state_dict())
        add("router", self.router.state_dict())
        add("linear_proj", self.linear_proj.state_dict())

        extra_path = save_path / ("wrapper.safetensors" if safe_serialization else "wrapper.bin")
        if safe_serialization:
            st.save_file(flat_sd, str(extra_path))
        else:
            torch.save(flat_sd, str(extra_path))

        cfg = {
            "class": "WhisperAccentedConditionalAdapters",
            "num_adapters": len(self.adapters),
            "bottleneck_size": self.adapters["adapter_0"].down.out_features if len(self.adapters) else None,
            "num_accents": self.num_accents,
            "accent_ignore_index": self.accent_ignore_index,
            "grl_lambda": self.grl_lambda,
        }
        (save_path / "wrapper_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, device: Optional[str] = None):
        ckpt = Path(ckpt_dir)
        base = WhisperForConditionalGeneration.from_pretrained(str(ckpt))

        cfg = json.loads((ckpt / "wrapper_config.json").read_text(encoding="utf-8"))
        model = cls(
            whisper_model=base,
            num_adapters=cfg["num_adapters"],
            bottleneck_size=cfg["bottleneck_size"],
            num_accents=cfg["num_accents"],
            accent_ignore_index=cfg["accent_ignore_index"],
            loss_weights=(1.0, 1.0, 2.0, 1.0),
            freeze_whisper=True,
            grl_lambda=cfg.get("grl_lambda", 1.0),
        )

        weights_path = ckpt / "wrapper.safetensors"
        if weights_path.exists():
            flat_sd = st.load_file(str(weights_path), device=device if device else "cpu")
        else:
            flat_sd = torch.load(str(ckpt / "wrapper.bin"), map_location=device if device else "cpu")

        def load_into(module, prefix):
            sd = {k.split(prefix + ".", 1)[1]: v for k, v in flat_sd.items() if k.startswith(prefix + ".")}
            module.load_state_dict(sd)

        load_into(model.head_accented, "head_accented")
        load_into(model.head_accent, "head_accent")
        load_into(model.head_gender, "head_gender")
        load_into(model.accent_embed, "accent_embed")
        load_into(model.cond_proj, "cond_proj")
        load_into(model.router, "router")
        load_into(model.linear_proj, "linear_proj")

        if "neutral_embed" in flat_sd:
            model.neutral_embed.data.copy_(flat_sd["neutral_embed"])

        adapters_sd = {k.split("adapters.", 1)[1]: v for k, v in flat_sd.items() if k.startswith("adapters.")}
        model.adapters.load_state_dict(adapters_sd)

        return model


# =============================================================================
# Custom Trainer
# =============================================================================

class MultiTaskSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        inputs = self._prepare_inputs(inputs)

        has_labels = "labels" in inputs and inputs["labels"] is not None
        labels = inputs.get("labels", None)

        if prediction_loss_only:
            with torch.no_grad():
                outputs = model(**inputs)
            loss = outputs["loss"].detach() if has_labels else None
            return (loss, None, labels)

        generated_tokens = None
        if self.args.predict_with_generate:
            gen_kwargs = {}
            if hasattr(self, "_gen_kwargs") and self._gen_kwargs is not None:
                gen_kwargs.update(self._gen_kwargs)

            if "max_length" not in gen_kwargs or gen_kwargs["max_length"] is None:
                gen_kwargs["max_length"] = self.args.generation_max_length
            if "num_beams" not in gen_kwargs or gen_kwargs["num_beams"] is None:
                gen_kwargs["num_beams"] = self.args.generation_num_beams

            with torch.no_grad():
                generated_tokens = model.generate(
                    input_features=inputs["input_features"],
                    attention_mask=inputs.get("attention_mask", None),
                    **gen_kwargs,
                )

            if generated_tokens is not None and hasattr(self, "_pad_tensors_to_max_len"):
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs["loss"].detach() if has_labels else None

        logits_acc = outputs["logits_accented_or_not"]
        logits_gender = outputs["logits_gender"]
        logits_accent = outputs["logits_accent"]

        lab_acc = inputs.get("accented_or_not_labels", None)
        lab_gender = inputs.get("gender_labels", None)
        lab_accent = inputs.get("accent_labels", None)

        lossvec_acc = lossvec_gender = lossvec_accent = None

        if lab_acc is not None:
            lossvec_acc = F.cross_entropy(logits_acc, lab_acc, reduction="none")

        if lab_gender is not None:
            lossvec_gender = F.cross_entropy(logits_gender, lab_gender, reduction="none")

        if lab_accent is not None:
            lossvec_accent = torch.zeros_like(lab_accent, dtype=torch.float32)
            mask_known = (lab_accent != int(model.accent_ignore_index))
            if mask_known.any():
                lossvec_accent[mask_known] = F.cross_entropy(
                    logits_accent[mask_known],
                    lab_accent[mask_known],
                    reduction="none"
                ).to(torch.float32)

        def make_2d(x):
            return x.unsqueeze(1) if (x is not None and x.dim() == 1) else x

        preds = (
            generated_tokens,
            logits_acc,
            logits_gender,
            logits_accent,
            make_2d(lab_acc),
            make_2d(lab_gender),
            make_2d(lab_accent),
            make_2d(lossvec_acc),
            make_2d(lossvec_gender),
            make_2d(lossvec_accent),
        )

        return (loss, preds, labels)


# =============================================================================
# External evaluation helper (reduces duplicated blocks)
# =============================================================================

def load_and_prepare_external_asr_only(
    ds_path: str,
    feature_extractor,
    tokenizer,
    id_map: Dict[str, str],
    text_col_candidates: List[str],
    logger: logging.Logger,
    lowercase: bool = True,
    filter_overlong: bool = False,
    max_label_len: Optional[int] = None,
):
    """
    Loads dataset from disk, selects {utt_id,audio,text}, normalizes column names.
    id_map: {existing_id_col: "utt_id"} (only one expected)
    text_col_candidates: possible transcript/text columns
    """
    raw = load_from_disk(ds_path)

    # Identify id column
    found_id = None
    for k in id_map.keys():
        if k in raw.column_names:
            found_id = k
            break
    if found_id is None:
        raise ValueError(f"No ID column found in {ds_path}. Expected one of: {list(id_map.keys())}")

    # Identify text column
    found_text = None
    for t in text_col_candidates:
        if t in raw.column_names:
            found_text = t
            break
    if found_text is None:
        raise ValueError(f"No text column found in {ds_path}. Expected one of: {text_col_candidates}")

    keep_cols = [found_id, "audio", found_text]
    raw = raw.select_columns(keep_cols).rename_columns({found_id: "utt_id", found_text: "text"})

    if lowercase:
        raw = raw.map(lambda b: {"text": (b["text"] or "").lower()})

    # Optional: filter long samples by token length (useful for some corpora)
    if filter_overlong:
        if max_label_len is None:
            raise ValueError("filter_overlong=True requires max_label_len.")
        def add_text_len(batch):
            ids = tokenizer(batch["text"]).input_ids
            return {"text_len": len(ids)}
        raw = raw.map(add_text_len)
        n_before = len(raw)
        raw = raw.filter(lambda x: x["text_len"] <= int(max_label_len))
        logger.info("Filtered %d overlong samples (kept %d / %d) with max_label_len=%d",
                    n_before - len(raw), len(raw), n_before, int(max_label_len))

    raw = raw.cast_column("audio", Audio(sampling_rate=16_000))

    cols = raw.column_names
    ds = raw.map(
        lambda b: prepare_dataset_asr_only(b, feature_extractor, tokenizer),
        remove_columns=[c for c in cols if c not in {"audio", "text", "utt_id", "text_len"}],
    )
    ds = ds.remove_columns([c for c in ds.column_names if c not in {"input_features", "labels"}])

    return raw, ds


# =============================================================================
# Training pipeline
# =============================================================================

def build_training_dataset(
    aesrc_train_dir: str,
    librispeech_dir: str,
    seed: int,
    eval_split_ratio: float,
    logger: logging.Logger,
) -> Tuple[DatasetDict, Dict[str, int]]:
    """
    Builds dataset with columns: audio, text, sex, accent, plus *_clf labels.
    Returns dataset dict {train, devel, test} and accent2id map.
    """
    # Load AESRC train shards (folder contains multiple split dirs)
    aesrc_train_splits = []
    train_cols_to_keep = ["audio", "transcript", "SEX", "accent"]

    for split_name in sorted(os.listdir(aesrc_train_dir)):
        split_path = os.path.join(aesrc_train_dir, split_name)
        if not os.path.isdir(split_path):
            continue
        ds_split = load_from_disk(split_path)
        ds_split = ds_split.select_columns(train_cols_to_keep)
        ds_split = ds_split.rename_columns({"SEX": "sex", "transcript": "text"})
        aesrc_train_splits.append(ds_split)

    if len(aesrc_train_splits) == 0:
        raise ValueError(f"No AESRC shards found under: {aesrc_train_dir}")

    aesrc_train = concatenate_datasets(aesrc_train_splits).map(normalize_sex)

    librispeech = load_from_disk(librispeech_dir)
    # Add unknown accent and drop typical LibriSpeech metadata columns if present
    librispeech = librispeech.add_column("accent", ["unknown"] * len(librispeech))
    for col in ["id", "speaker_id", "chapter_id"]:
        if col in librispeech.column_names:
            librispeech = librispeech.remove_columns([col])

    dataset_train = concatenate_datasets([aesrc_train, librispeech])

    dataset_train = dataset_train.map(lambda x: {"accented_or_not_clf": 0 if x["accent"] == "unknown" else 1})
    dataset_train = dataset_train.map(lambda x: {"gender_clf": 0 if x["sex"] == "female" else 1})

    # NOTE: Keep your mapping, but it’s not personally identifying.
    accent2id = {
        "unknown": 0,
        "American English Speech Data": 1,
        "British English Speech Data": 2,
        "Canadian English Speech Data": 3,
        "Chinese Speaking English Speech Data": 4,
        "Indian English Speech Data": 5,
        "Japanese Speaking English Speech Data": 6,
        "Korean Speaking English Speech Data": 7,
        "Portuguese Speaking English Speech Data": 8,
        "Russian Speaking English Speech Data": 9,
        "Spanish Speaking English Speech Data": 10,
    }
    dataset_train = dataset_train.map(lambda x: {"accent_clf": accent2id[str(x["accent"])]})

    dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16_000))
    logger.info("Train rows: %d | cols=%s", len(dataset_train), list(dataset_train.features.keys()))

    # Train/dev/test split
    if eval_split_ratio > 0.0:
        try:
            split = dataset_train.train_test_split(
                test_size=eval_split_ratio,
                seed=seed,
                stratify_by_column="accent_clf",
            )
        except Exception:
            split = dataset_train.train_test_split(test_size=eval_split_ratio, seed=seed)

        devel_split = split["test"].shuffle(seed=seed)
        # keep your “subset dev” behavior
        devel_split = devel_split.shuffle(seed=seed).select(range(min(1000, len(devel_split))))

        dataset = DatasetDict({"train": split["train"], "devel": devel_split, "test": split["test"]})
    else:
        dataset_train = dataset_train.shuffle(seed)
        dataset = DatasetDict({"train": dataset_train, "devel": dataset_train, "test": dataset_train})

    dataset = dataset.map(lambda b: {"text": (b["text"] or "").lower()})
    return dataset, accent2id


def tokenize_and_prepare_splits(dataset: DatasetDict, feature_extractor, tokenizer, logger: logging.Logger) -> DatasetDict:
    keep_after_map = {"input_features", "labels", "accented_or_not_labels", "gender_labels", "accent_labels"}
    for split_name in dataset.keys():
        cols = dataset[split_name].column_names
        dataset[split_name] = dataset[split_name].map(
            lambda b: prepare_dataset(b, feature_extractor, tokenizer),
            remove_columns=[c for c in cols if c not in {"audio", "text", "accented_or_not_clf", "gender_clf", "accent_clf"}],
        )
        cols2 = dataset[split_name].column_names
        drop2 = [c for c in cols2 if c not in keep_after_map]
        if drop2:
            dataset[split_name] = dataset[split_name].remove_columns(drop2)

    logger.info("After map:")
    for split_name in dataset.keys():
        logger.info("%s: %d rows | cols=%s", split_name, len(dataset[split_name]), dataset[split_name].column_names)
    return dataset


def build_model_and_processor(args) -> Tuple[WhisperAccentedConditionalAdapters, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor]:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)

    base_whisper = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    model = WhisperAccentedConditionalAdapters(
        whisper_model=base_whisper,
        num_adapters=args.num_adapters,
        bottleneck_size=args.bottleneck_size,
        num_accents=args.num_accents,
        accent_ignore_index=args.accent_ignore_index,
        freeze_whisper=True,
        grl_lambda=args.grl_lambda,
        loss_weights=(1.0, 1.0, 2.0, 1.0),
    )

    # Avoid tying issues when wrapped
    model.whisper.tie_weights = lambda: None
    model.whisper.proj_out.weight = nn.Parameter(model.whisper.proj_out.weight.clone().detach())

    model.whisper.generation_config.language = args.language
    model.whisper.generation_config.task = args.task
    model.whisper.generation_config.forced_decoder_ids = None
    model.whisper.config.use_cache = False
    model.whisper.config.pad_token_id = tokenizer.pad_token_id

    return model, processor, tokenizer, feature_extractor


def build_training_args(args, output_dir: str) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=1,
        report_to=(["tensorboard"] if args.report_to_tensorboard else []),
    )


def save_last_checkpoint(
    model: WhisperAccentedConditionalAdapters,
    processor: WhisperProcessor,
    tokenizer: WhisperTokenizer,
    feature_extractor: WhisperFeatureExtractor,
    training_args: Seq2SeqTrainingArguments,
    trainer: Seq2SeqTrainer,
    out_dir: str,
    logger: logging.Logger,
) -> None:
    last_dir = Path(out_dir) / "last_checkpoint"
    last_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(last_dir))
    processor.save_pretrained(str(last_dir))
    tokenizer.save_pretrained(str(last_dir))
    feature_extractor.save_pretrained(str(last_dir))
    (last_dir / "training_args.json").write_text(training_args.to_json_string(), encoding="utf-8")
    trainer.state.save_to_json(str(last_dir / "trainer_state.json"))

    logger.info("Saved last checkpoint to: %s", str(last_dir))


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MoAA training + evaluation runner (anonymized).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    p.add_argument("--aesrc_train_dir", type=str, required=True, help="Directory containing AESRC train shard folders.")
    p.add_argument("--librispeech_dir", type=str, required=True, help="Path to LibriSpeech train-clean-100 HF dataset dir.")

    # Optional external tests
    p.add_argument("--aesrc_test_dir", type=str, default=None, help="Optional AESRC test HF dataset dir (ASR-only eval).")
    p.add_argument("--openslr_test_dir", type=str, default=None, help="Optional OpenSLR test HF dataset dir.")
    p.add_argument("--edacc_test_dir", type=str, default=None, help="Optional EdAcc test HF dataset dir.")
    p.add_argument("--globe_test_dir", type=str, default=None, help="Optional GLOBE test HF dataset dir.")

    # Model
    p.add_argument("--model_name", type=str, default="openai/whisper-small")
    p.add_argument("--language", type=str, default="en")
    p.add_argument("--task", type=str, default="transcribe")

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_false", dest="fp16")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # MoAA
    p.add_argument("--num_adapters", type=int, default=10)
    p.add_argument("--bottleneck_size", type=int, default=192)
    p.add_argument("--num_accents", type=int, default=10)
    p.add_argument("--accent_ignore_index", type=int, default=-100)
    p.add_argument("--grl_lambda", type=float, default=1.0)

    # Split behavior
    p.add_argument("--eval_split_ratio", type=float, default=0.1)

    # Generation / reporting
    p.add_argument("--generation_max_length", type=int, default=225)
    p.add_argument("--report_to_tensorboard", action="store_true", default=True)
    p.add_argument("--no_tensorboard", action="store_false", dest="report_to_tensorboard")

    # Output
    p.add_argument("--output_root", type=str, required=True, help="Root directory for outputs (checkpoints/logs/csv).")
    p.add_argument("--run_name", type=str, default="moaa_run", help="Run name (subfolder under output_root).")

    return p


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.output_root) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    csv_dir = run_dir / "csv"

    for d in [ckpt_dir, log_dir, csv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_path=str(log_dir / "run.log"))
    logger.info("Run directory: %s", str(run_dir))
    logger.info("Args: %s", json.dumps(vars(args), indent=2))
    check_gpu(logger)
    set_seed(args.seed)

    # ---------------- Load and build dataset ----------------
    dataset, _accent2id = build_training_dataset(
        aesrc_train_dir=args.aesrc_train_dir,
        librispeech_dir=args.librispeech_dir,
        seed=args.seed,
        eval_split_ratio=args.eval_split_ratio,
        logger=logger,
    )

    # ---------------- Feature extraction/tokenization ----------------
    model, processor, tokenizer, feature_extractor = build_model_and_processor(args)

    dataset = tokenize_and_prepare_splits(dataset, feature_extractor, tokenizer, logger)

    # Save raw internal test for CSV saving (needs multitask labels present)
    internal_test_raw = dataset["test"]

    data_collator = DataCollatorSpeechSeq2SeqWithPaddingAndClf(
        processor=processor,
        decoder_start_token_id=model.whisper.config.decoder_start_token_id,
    )

    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    # ---------------- Training args & trainer ----------------
    training_args = build_training_args(args, output_dir=str(ckpt_dir))

    trainer = MultiTaskSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["devel"],
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics_asr_and_clf,
            tokenizer=tokenizer,
            metric_wer=metric_wer,
            metric_cer=metric_cer,
            accent_ignore_index=args.accent_ignore_index,
        ),
        tokenizer=processor.tokenizer,
    )

    # ---------------- Eval before training ----------------
    logger.info("*" * 60)
    logger.info("Internal TEST evaluation (BEFORE TRAINING)")
    pre = trainer.predict(dataset["test"])
    logger.info("WER: %.4f | CER: %.4f", pre.metrics.get("test_wer", float("nan")), pre.metrics.get("test_cer", float("nan")))
    logger.info("*" * 60)

    # ---------------- Train ----------------
    trainer.train()
    logger.info("Training complete.")

    # ---------------- Eval after training ----------------
    logger.info("*" * 60)
    logger.info("Internal TEST evaluation (AFTER TRAINING)")
    post = trainer.predict(dataset["test"])
    logger.info("WER: %.4f | CER: %.4f", post.metrics.get("test_wer", float("nan")), post.metrics.get("test_cer", float("nan")))
    logger.info("*" * 60)

    # ---------------- Save internal predictions ----------------
    save_predictions_csv_multitask(
        pred_output=post,
        raw_ds=internal_test_raw,
        tokenizer=tokenizer,
        csv_path=str(csv_dir / "internal_test_multitask.csv"),
    )

    # ---------------- External ASR-only evaluations ----------------
    def run_external_eval(name: str, ds_path: str, id_keys: List[str], text_keys: List[str], filter_overlong: bool = False):
        logger.info("*" * 60)
        logger.info("External eval: %s | path=%s", name, ds_path)

        max_label_len = model.whisper.config.max_target_positions if filter_overlong else None

        raw, ds = load_and_prepare_external_asr_only(
            ds_path=ds_path,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            id_map={k: "utt_id" for k in id_keys},
            text_col_candidates=text_keys,
            logger=logger,
            lowercase=True,
            filter_overlong=filter_overlong,
            max_label_len=max_label_len,
        )

        collator = DataCollatorSpeechSeq2SeqWithPaddingASROnly(
            processor=processor,
            decoder_start_token_id=model.whisper.config.decoder_start_token_id,
        )

        ext_trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            data_collator=collator,
            compute_metrics=partial(
                compute_metrics_asr_only,
                tokenizer=tokenizer,
                metric_wer=metric_wer,
                metric_cer=metric_cer,
            ),
            tokenizer=processor.tokenizer,
        )

        res = ext_trainer.predict(ds)
        logger.info("%s WER: %.4f | CER: %.4f", name, res.metrics.get("test_wer", float("nan")), res.metrics.get("test_cer", float("nan")))

        save_predictions_csv_external_with_clf(
            pred_output=res,
            trainer=ext_trainer,
            dataset=ds,
            raw_ds=raw,
            tokenizer=tokenizer,
            csv_path=str(csv_dir / f"{name.lower()}_asr_only_with_clf.csv"),
            id_key="utt_id",
        )

        logger.info("*" * 60)

    if args.aesrc_test_dir:
        run_external_eval(
            name="AESRC",
            ds_path=args.aesrc_test_dir,
            id_keys=["utt_id"],
            text_keys=["transcript", "text"],
            filter_overlong=False,
        )

    if args.openslr_test_dir:
        run_external_eval(
            name="OpenSLR",
            ds_path=args.openslr_test_dir,
            id_keys=["line_id", "utt_id"],
            text_keys=["transcript", "text"],
            filter_overlong=False,
        )

    if args.edacc_test_dir:
        # keep your previous behavior: filter long labels
        run_external_eval(
            name="EdAcc",
            ds_path=args.edacc_test_dir,
            id_keys=["speaker", "utt_id"],
            text_keys=["transcript", "text"],
            filter_overlong=True,
        )

    if args.globe_test_dir:
        run_external_eval(
            name="GLOBE",
            ds_path=args.globe_test_dir,
            id_keys=["speaker_id", "utt_id"],
            text_keys=["transcript", "text"],
            filter_overlong=False,
        )

    # ---------------- Save final checkpoint bundle ----------------
    save_last_checkpoint(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        training_args=training_args,
        trainer=trainer,
        out_dir=str(ckpt_dir),
        logger=logger,
    )



#############################################################################
#############################################################################

if __name__ == "__main__":
    main()




