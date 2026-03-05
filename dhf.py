"""
Deterministic Hallucination Filtering (DHF) on ASR outputs.

Outputs written to --out_dir:
  - asr_hallucination_analysis.csv              (insertion analysis result; if enabled)
  - dhf_with_cleanup.csv                        (out_df + DHF columns)
  - dhf_with_cleanup_filtered.csv               (optional filtered version)
  - eval_metrics.json                           (if eval enabled)
  - inserted_word_freq_topK.txt                 (if insertion analysis enabled)

Reference-free example (no references required):
  python dhf_pipeline.py \
    --csv_path /path/to/file.csv \
    --pred_col pred_text \
    --out_dir dhf_outputs/ref_free_run1 \
    --no_insertion_analysis \
    --no_eval \
    --filter_numeric_only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import jiwer
    from jiwer import process_words
except Exception:
    jiwer = None
    process_words = None


# ============================================================
# I/O helpers
# ============================================================

def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# Stage 1: insertion analysis (reference-based) [optional]
# ============================================================

def extract_inserted_words(out) -> List[str]:
    """
    Compatible across jiwer versions where:
      - out.alignments is List[List[AlignmentChunk]]
      - or List[SentenceOutput] with .operations
    """
    inserted: List[str] = []
    alignments = getattr(out, "alignments", []) or []

    for sent in alignments:
        if hasattr(sent, "operations"):
            chunks = sent.operations
        elif isinstance(sent, list):
            chunks = sent
        else:
            chunks = [sent]

        for ch in chunks:
            if isinstance(ch, dict):
                if ch.get("type") == "insert":
                    hw = ch.get("hypothesis_words", None)
                    if hw is None:
                        hw = ch.get("hypothesis_word", None)
                    if isinstance(hw, list):
                        inserted.extend(hw)
                    elif hw:
                        inserted.append(hw)
                continue

            ch_type = getattr(ch, "type", None)
            if ch_type == "insert":
                hw = getattr(ch, "hypothesis_words", None)
                if hw is None:
                    hw = getattr(ch, "hypothesis_word", None)
                if isinstance(hw, list):
                    inserted.extend(hw)
                elif hw:
                    inserted.append(hw)

    return inserted


def run_insertion_analysis(
    df: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    topk_words: int = 30,
) -> Tuple[pd.DataFrame, Counter]:
    if process_words is None:
        raise RuntimeError("jiwer is required for insertion analysis. Please install/import jiwer.")

    pred_norm = df[pred_col].fillna("").astype(str).map(normalize_text)
    ref_norm  = df[ref_col].fillna("").astype(str).map(normalize_text)

    rows = []
    all_inserted: List[str] = []

    for i, (hyp, ref) in enumerate(zip(pred_norm, ref_norm)):
        out = process_words(ref, hyp)  # ref first, hyp second

        ins = out.insertions
        dele = out.deletions
        sub = out.substitutions
        ref_len = len(out.references)
        hyp_len = len(out.hypotheses)

        wer = (ins + dele + sub) / ref_len if ref_len > 0 else 0.0

        inserted_words = extract_inserted_words(out)
        all_inserted.extend(inserted_words)

        rows.append({
            "row": i,
            "ref_len": ref_len,
            "hyp_len": hyp_len,
            "wer": wer,
            "ins": ins,
            "del": dele,
            "sub": sub,
            "hallucinated_words": " ".join(inserted_words),
            "hall_rate_ref": (ins / ref_len) if ref_len > 0 else 0.0,
            "hall_rate_hyp": (ins / hyp_len) if hyp_len > 0 else 0.0,
        })

    hall = pd.DataFrame(rows)
    freq = Counter([w for w in all_inserted if w and w.strip()])

    print("\nMost frequently inserted (hallucinated) words:")
    for w, n in freq.most_common(topk_words):
        print(f"{w:>15}  {n}")

    return hall, freq


# ============================================================
# Stage 2: fully reference-free cleanup (DHF)
# ============================================================

NUM_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",
    "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety",
    "hundred","thousand","million","billion","trillion"
}

SPAM_TOKENS = {
    "sil", "sp", "uh", "um", "erm", "eh", "ah", "mm", "hmm",
    "noise", "background", "static"
}

HAS_ALPHA = re.compile(r"[a-z]", re.I)
NUMLIKE = re.compile(
    r"(" 
    r"\d{1,2}:\d{2}(?::\d{2})?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
    r"|\d+(?:st|nd|rd|th)"
    r")",
    re.I
)

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _pre_clean_text(s: str) -> str:
    s = _normalize_ws((s or "").lower())
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    return s

def _tokens(s: str) -> List[str]:
    s = _pre_clean_text(s)
    return re.findall(r"[a-z]+|\d+(?:st|nd|rd|th)?", s)

def _detok(toks: List[str]) -> str:
    return " ".join(toks).strip()

def _is_ordinal(t: str) -> bool:
    return bool(re.fullmatch(r"\d+(st|nd|rd|th)", t))

def _is_numberish(t: str) -> bool:
    return (t.isdigit() or t in NUM_WORDS or _is_ordinal(t))

def flag_number_only_text(s) -> int:
    s = "" if s is None else str(s).strip()
    if not s:
        return 0
    if HAS_ALPHA.search(s):
        return 0
    return 1 if NUMLIKE.search(s) else 0

def drop_apostrophe_garbage(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return raw
    s = raw.strip()
    if s.count("'") >= 12 and s.count(" ") <= 2:
        return ""
    return raw

def compress_token_runs(toks: List[str], keep_max: int = 2) -> List[str]:
    if not toks:
        return toks
    out = [toks[0]]
    run = 1
    for t in toks[1:]:
        if t == out[-1]:
            run += 1
            if run <= keep_max:
                out.append(t)
        else:
            run = 1
            out.append(t)
    return out

def number_spam_ratio(toks: List[str]) -> float:
    if not toks:
        return 0.0
    return sum(1 for t in toks if _is_numberish(t)) / len(toks)

def _is_lalalala_spam_token(t: str) -> bool:
    if not t or not t.isalpha():
        return False
    if len(t) < 12:
        return False
    if len(set(t)) > 5:
        return False
    if re.search(r"(?:la){5,}", t) or re.search(r"(?:al){5,}", t) or re.search(r"(?:ala){4,}", t):
        return True
    if t.startswith("sil") and (re.search(r"(?:la){4,}", t[3:]) or re.search(r"(?:ala){3,}", t[3:])):
        return True
    return False

def remove_noise_tokens(toks: List[str]) -> List[str]:
    out = []
    for t in toks:
        if t in SPAM_TOKENS:
            continue
        if len(t) == 1 and t.isalpha():
            continue
        if _is_lalalala_spam_token(t):
            continue
        out.append(t)
    return out

def find_best_consecutive_repeat(tokens: List[str], min_k=2, max_k=16, min_reps=3):
    n = len(tokens)
    best = None
    for start in range(n):
        remaining = n - start
        max_k_eff = min(max_k, remaining // min_reps)
        for k in range(min_k, max_k_eff + 1):
            block = tokens[start:start+k]
            reps = 1
            i = start + k
            while i + k <= n and tokens[i:i+k] == block:
                reps += 1
                i += k
            if reps >= min_reps:
                covered = k * reps
                cand = (covered, k, -start, start, reps)
                if best is None or cand > best:
                    best = cand
    if best is None:
        return None
    _, k, _, start, reps = best
    return start, k, reps

def compress_best_repeat_block(tokens: List[str]) -> List[str]:
    best = find_best_consecutive_repeat(tokens, min_k=2, max_k=16, min_reps=3)
    if best is None:
        return tokens
    start, k, _reps = best
    return tokens[:start] + tokens[start:start+k]

def cut_at_second_occurrence(tokens: List[str], Ls=(10, 9, 8, 7, 6, 5, 4, 3)) -> List[str]:
    n = len(tokens)
    for L in Ls:
        seen = {}
        for i in range(0, n - L + 1):
            ph = tuple(tokens[i:i+L])
            if ph in seen:
                return tokens[:i]
            seen[ph] = i
    return tokens

def cut_number_word_tail(tokens: List[str], trigger_words=("thousand", "hundred", "million", "billion", "trillion")) -> List[str]:
    if len(tokens) < 12:
        return tokens
    for i in range(10, len(tokens)):
        window = tokens[i-10:i]
        num_ratio = number_spam_ratio(window)
        trig_cnt = sum(1 for t in window if t in trigger_words)
        if num_ratio >= 0.60 or trig_cnt >= 4:
            cut = max(3, i - 10)
            return tokens[:cut]
    return tokens

def generate_candidates(pred_text: str) -> List[str]:
    raw = pred_text or ""
    raw = drop_apostrophe_garbage(raw)
    if raw == "":
        return [""]

    toks0 = _tokens(raw)
    if not toks0:
        return [""]

    toks = remove_noise_tokens(toks0)
    toks = compress_token_runs(toks, keep_max=2)
    if not toks:
        return [""]

    cands = [
        _detok(toks),
        _detok(cut_number_word_tail(toks)),
        _detok(compress_best_repeat_block(toks)),
        _detok(cut_at_second_occurrence(toks)),
    ]

    if len(toks) >= 25 and number_spam_ratio(toks) >= 0.70:
        cands.append("")

    seen = set()
    uniq = []
    for c in cands:
        c = _normalize_ws(c)
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def longest_token_run(toks: List[str]) -> int:
    if not toks:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def repeated_ngram_coverage(tokens: List[str], n=3) -> int:
    if len(tokens) < 2 * n:
        return 0
    seen = {}
    covered = 0
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        if ng in seen:
            covered += n
        else:
            seen[ng] = i
    return covered

def reference_free_quality_score(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return -100.0

    L = len(toks)
    uniq_ratio = len(set(toks)) / max(1, L)
    num_ratio = number_spam_ratio(toks)
    spam_cnt = sum(1 for t in toks if t in SPAM_TOKENS or _is_lalalala_spam_token(t))
    single_char_cnt = sum(1 for t in toks if len(t) == 1 and t.isalpha())
    long_run = longest_token_run(toks)
    rep3 = repeated_ngram_coverage(toks, n=3)
    rep4 = repeated_ngram_coverage(toks, n=4)
    num_only = flag_number_only_text(text)

    score = 0.0
    score -= 0.03 * max(0, L - 40)
    score -= 2.0 * spam_cnt
    score -= 1.5 * single_char_cnt
    score -= 2.0 * max(0, long_run - 2)
    score -= 0.25 * rep3
    score -= 0.35 * rep4
    score -= 8.0 * max(0, num_ratio - 0.5)
    score -= 3.0 * num_only
    score += 2.0 * uniq_ratio

    if L <= 1:
        score -= 20.0
    elif L <= 2:
        score -= 8.0

    return score

def is_suspicious_hypothesis(pred_text: str) -> bool:
    toks = _tokens(pred_text)
    if not toks:
        return False

    L = len(toks)
    num_ratio = number_spam_ratio(toks)
    spam_cnt = sum(1 for t in toks if t in SPAM_TOKENS or _is_lalalala_spam_token(t))
    single_char_cnt = sum(1 for t in toks if len(t) == 1 and t.isalpha())
    long_run = longest_token_run(toks)
    rep3 = repeated_ngram_coverage(toks, n=3)
    uniq_ratio = len(set(toks)) / max(1, L)
    pred_num_only = flag_number_only_text(pred_text)

    if spam_cnt >= 1:
        return True
    if single_char_cnt >= 4:
        return True
    if long_run >= 4:
        return True
    if rep3 >= 6:
        return True
    if L >= 25 and num_ratio >= 0.70:
        return True
    if L >= 20 and uniq_ratio < 0.35:
        return True
    if pred_num_only == 1 and L >= 3:
        return True

    return False

def pick_best_candidate_reference_free(pred_text: str, margin: float = 1.0) -> str:
    orig = _normalize_ws(pred_text or "")
    candidates = generate_candidates(pred_text)

    if orig not in candidates:
        candidates = [orig] + candidates

    scored = []
    for cand in candidates:
        score = reference_free_quality_score(cand)
        tok_len = len(_tokens(cand))
        scored.append((score, tok_len, cand))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, _, best_cand = scored[0]
    orig_score = reference_free_quality_score(orig)

    if best_cand != orig and best_score >= orig_score + margin:
        return best_cand
    return orig


# ============================================================
# Stage 3: evaluation (reference-only reporting) [optional]
# ============================================================

def evaluate_asr_df(df: pd.DataFrame, hyp_col: str, ref_col: str, title: str) -> Dict[str, float]:
    if jiwer is None:
        raise RuntimeError("jiwer is required for evaluation. Please install/import jiwer.")

    refs = df[ref_col].fillna("").astype(str).tolist()
    hyps = df[hyp_col].fillna("").astype(str).tolist()

    w_out = jiwer.process_words(refs, hyps)
    c_out = jiwer.process_characters(refs, hyps)

    print(f"\n=== {title} ===")
    print("WORD LEVEL")
    print(f"WER: {w_out.wer:.6f}")
    print(
        f"S/I/D/H: {w_out.substitutions}/{w_out.insertions}/{w_out.deletions}/{w_out.hits} "
        f"(N_ref_words={w_out.hits + w_out.substitutions + w_out.deletions})"
    )

    print("\nCHAR LEVEL")
    print(f"CER: {c_out.cer:.6f}")
    print(
        f"S/I/D/H: {c_out.substitutions}/{c_out.insertions}/{c_out.deletions}/{c_out.hits} "
        f"(N_ref_chars={c_out.hits + c_out.substitutions + c_out.deletions})"
    )

    return {"wer": float(w_out.wer), "cer": float(c_out.cer)}


# ============================================================
# CLI / main
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DHF pipeline runner (optional insertion analysis + reference-free cleanup + optional eval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--csv_path", type=str, required=True, help="Input CSV path.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory to write CSVs and logs.")

    p.add_argument("--pred_col", type=str, default="pred_text", help="Prediction column name.")
    p.add_argument("--ref_col", type=str, default="ref_text", help="Reference column name.")

    p.add_argument("--do_insertion_analysis", action="store_true", default=True, help="Run reference-based insertion analysis.")
    p.add_argument("--no_insertion_analysis", action="store_false", dest="do_insertion_analysis")

    p.add_argument("--do_eval", action="store_true", default=True, help="Run reference-only evaluation (WER/CER reporting).")
    p.add_argument("--no_eval", action="store_false", dest="do_eval")

    p.add_argument("--filter_numeric_only", action="store_true", default=True, help="Drop rows where cleaned pred is numeric-only.")
    p.add_argument("--no_filter_numeric_only", action="store_false", dest="filter_numeric_only")

    p.add_argument("--margin", type=float, default=1.0, help="Score margin required to replace original with cleaned candidate.")
    p.add_argument("--topk_insert_words", type=int, default=30, help="Top-K inserted words to print/save.")

    p.add_argument("--sep", type=str, default=",", help="CSV separator.")
    p.add_argument("--encoding", type=str, default="utf-8", help="CSV encoding.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    mkdir(out_dir)
    save_json(vars(args), out_dir / "run_args.json")

    df = pd.read_csv(args.csv_path, sep=args.sep, encoding=args.encoding)
    if args.pred_col not in df.columns:
        raise ValueError(f"pred_col '{args.pred_col}' not found. Columns: {list(df.columns)}")

    has_ref = args.ref_col in df.columns

    # -------------------------
    # insertion analysis => out_df (optional)
    # -------------------------
    if args.do_insertion_analysis:
        if not has_ref:
            print(f"[WARN] ref_col '{args.ref_col}' not found; skipping insertion analysis.")
            out_df = df.copy()
        else:
            hall, freq = run_insertion_analysis(df, args.pred_col, args.ref_col, topk_words=args.topk_insert_words)
            out_df = pd.concat([df[[args.pred_col, args.ref_col]].reset_index(drop=True), hall.reset_index(drop=True)], axis=1)

            out_path = out_dir / "asr_hallucination_analysis.csv"
            out_df.to_csv(out_path, index=False)
            print(f"\nSaved: {out_path}")

            freq_path = out_dir / f"inserted_word_freq_top{args.topk_insert_words}.txt"
            with freq_path.open("w", encoding="utf-8") as f:
                for w, n in freq.most_common(args.topk_insert_words):
                    f.write(f"{w}\t{n}\n")
            print(f"Saved: {freq_path}")
    else:
        out_df = df.copy()

    # normalize naming for downstream DHF stage
    work = out_df.copy()
    if args.pred_col != "pred_text":
        work = work.rename(columns={args.pred_col: "pred_text"})
    if has_ref and args.ref_col != "ref_text":
        work = work.rename(columns={args.ref_col: "ref_text"})

    # -------------------------
    # reference-free hallucination suppression
    # -------------------------
    work["sus_ref_free"] = work["pred_text"].fillna("").astype(str).apply(is_suspicious_hypothesis)
    work["pred_number_only"] = work["pred_text"].fillna("").astype(str).apply(flag_number_only_text)

    work["pred_text_final"] = work["pred_text"].fillna("").astype(str)
    mask = work["sus_ref_free"]
    work.loc[mask, "pred_text_final"] = (
        work.loc[mask, "pred_text"]
        .fillna("")
        .astype(str)
        .apply(lambda s: pick_best_candidate_reference_free(s, margin=args.margin))
    )

    work["pred_final_number_only"] = work["pred_text_final"].fillna("").astype(str).apply(flag_number_only_text)

    if args.filter_numeric_only:
        out_df_filtered = work.loc[work["pred_final_number_only"] != 1].copy()
    else:
        out_df_filtered = work.copy()

    print("\n=== Cleanup Summary ===")
    print("Total rows:", len(work))
    print("Suspicious rows:", int(work["sus_ref_free"].sum()))
    print("Original pred_number_only rows:", int(work["pred_number_only"].sum()))
    print("Final pred_final_number_only rows:", int(work["pred_final_number_only"].sum()))
    print("Remaining after numeric-only filtering:", len(out_df_filtered))

    inspect_cols = [c for c in ["row", "pred_text", "pred_text_final", "sus_ref_free", "pred_number_only", "pred_final_number_only"] if c in work.columns]
    print("\nSample suspicious rows:")
    if int(work["sus_ref_free"].sum()) > 0:
        print(work.loc[work["sus_ref_free"], inspect_cols].head(30).to_string(index=False))
    else:
        print("(none)")

    out_clean = out_dir / "dhf_with_cleanup.csv"
    out_filt = out_dir / "dhf_with_cleanup_filtered.csv"
    work.to_csv(out_clean, index=False)
    out_df_filtered.to_csv(out_filt, index=False)
    print(f"\nSaved: {out_clean}")
    print(f"Saved: {out_filt}")

    # -------------------------
    # optional evaluation (reference-only)
    # -------------------------
    if args.do_eval:
        if not has_ref:
            print(f"[WARN] ref_col '{args.ref_col}' not found; skipping evaluation.")
        else:
            if jiwer is None:
                print("[WARN] jiwer not available; skipping evaluation.")
            else:
                metrics = {}
                metrics["before_cleanup"] = evaluate_asr_df(work, hyp_col="pred_text", ref_col="ref_text", title="Before Cleanup")
                metrics["after_cleanup"] = evaluate_asr_df(work, hyp_col="pred_text_final", ref_col="ref_text", title="After Reference-Free Cleanup")
                metrics["after_cleanup_filtered"] = evaluate_asr_df(out_df_filtered, hyp_col="pred_text_final", ref_col="ref_text", title="After Cleanup + Numeric-only Filtering")
                save_json(metrics, out_dir / "eval_metrics.json")
                print(f"\nSaved: {out_dir / 'eval_metrics.json'}")

    print("\n[DONE]")


##############################################################
##############################################################

if __name__ == "__main__":
    main()



