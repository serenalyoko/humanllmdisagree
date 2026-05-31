"""
Zuobiao Ideology Calculator
============================
Computes Pan & Xu (2018) ideology scores from LLM pipeline output.

Matches questions to CFA loadings by parsing the "Qxx." prefix at the
start of each question string — so file order does not matter.

Three scores are produced:
  lt1  — Political Liberalism / Individual Freedom   (+ = more liberal)
  lt2  — Promarket / Non-traditional Values          (+ = more promarket)
  lt3  — Nationalism × (−1)                         (+ = less nationalist)

Answer scoring (same as formula_calculator.py):
  "strongly disagree" → -2
  "disagree"          → -1
  "neutral"           →  0  (skipped)
  "agree"             → +1
  "strongly agree"    → +2

Scores are normalized to [−10, +10].

Usage
-----
Standalone:
    python zuobiao_calculator.py \
        --input  data/pc_results/claude_pc_results.json \
        --loadings output/cfa_loadings.csv \
        --output data/pc_results/claude_zuobiao_scores.json

As a module:
    from zuobiao_calculator import scores_from_pipeline_output
    scores = scores_from_pipeline_output(pipeline_output_dict)
"""

import re
import json
import csv
import argparse
import pathlib
import statistics
from typing import Optional

# ── CFA loadings  { question_number (int) -> {"loading": float, "factor": str} }
_LOADINGS: dict[int, dict] = {}


def _load_loadings(csv_path: str = "./data/cfa_loadings.csv") -> None:
    """Read cfa_loadings.csv and populate _LOADINGS."""
    global _LOADINGS
    path = pathlib.Path(csv_path)
    if not path.exists():
        for candidate in [
            "./data/cfa_loadings.csv",
            "./data/cfa_loadings.csv",
            pathlib.Path(__file__).parent / "./data/cfa_loadings.csv",
            pathlib.Path(__file__).parent / "output" / "./data/cfa_loadings.csv",
        ]:
            if pathlib.Path(candidate).exists():
                path = pathlib.Path(candidate)
                break
        else:
            raise FileNotFoundError(
                "cfa_loadings.csv not found. Generate it from R:\n"
                "  factor_map <- data.frame(\n"
                "    question = questions,\n"
                "    loading  = as.numeric(coef),\n"
                "    factor   = NA\n"
                "  )\n"
                "  factor_map$factor[modelA[[1]]] <- 'lt1'\n"
                "  factor_map$factor[modelA[[2]]] <- 'lt2'\n"
                "  factor_map$factor[modelA[[3]]] <- 'lt3'\n"
                "  write.csv(factor_map, 'output/cfa_loadings.csv', row.names=FALSE)"
            )
    _LOADINGS = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # accepts "q1", "Q1", "q26", "Q26", etc.
            qnum = int(re.sub(r"[^0-9]", "", row["question"]))
            _LOADINGS[qnum] = {
                "loading": float(row["loading"]),
                "factor":  row["factor"].strip(),
            }


try:
    _load_loadings()
except FileNotFoundError:
    pass   # will raise clearly when compute is called


# ── Parse Q-number from question string ───────────────────────────────────────
_QNUM_RE = re.compile(r"^Q(\d+)\.", re.IGNORECASE)

def parse_qnum(question_text: str) -> Optional[int]:
    """
    Extract the question number from a string like 'Q26. Some text...'.
    Returns None if the prefix is missing or unrecognisable.
    """
    m = _QNUM_RE.match(question_text.strip())
    return int(m.group(1)) if m else None


# ── Answer → numeric score ────────────────────────────────────────────────────
ANSWER_TO_SCORE = {
    "strongly disagree": -2,
    "disagree":          -1,
    "neutral":            0,
    "agree":             +1,
    "strongly agree":    +2,
}


# ── Core scoring ──────────────────────────────────────────────────────────────
def compute_zuobiao_scores(
    questions: list[str],
    answers:   list[Optional[str]],
) -> dict:
    """
    Compute lt1, lt2, lt3 scores.

    Args:
        questions : List of question strings, each starting with "Qxx."
                    (file order does not matter — Q-number is parsed from text).
        answers   : Parallel list of answer strings or None.
                    Each element: "strongly disagree" | "disagree" | "neutral"
                                | "agree" | "strongly agree" | None

    Returns:
        dict with lt1, lt2, lt3 (float, −10 to +10) plus diagnostic fields.
    """
    if not _LOADINGS:
        _load_loadings()

    if len(questions) != len(answers):
        raise ValueError(
            f"questions ({len(questions)}) and answers ({len(answers)}) must be the same length."
        )

    raw   = {"lt1": 0.0, "lt2": 0.0, "lt3": 0.0}
    max_w = {"lt1": 0.0, "lt2": 0.0, "lt3": 0.0}
    count = {"lt1": 0,   "lt2": 0,   "lt3": 0}
    unmatched = []
    answered  = 0

    for question, answer in zip(questions, answers):
        if answer is None:
            continue
        score = ANSWER_TO_SCORE.get(answer.strip().lower())
        if score is None or score == 0:
            continue   # neutral or unrecognised → skip

        qnum = parse_qnum(question)
        if qnum is None:
            unmatched.append(question[:60])
            continue

        info = _LOADINGS.get(qnum)
        if info is None:
            unmatched.append(f"Q{qnum} (not in cfa_loadings)")
            continue

        factor  = info["factor"]
        loading = info["loading"]

        raw[factor]   += loading * score
        max_w[factor] += abs(loading) * 2
        count[factor] += 1
        answered += 1

    def normalize(raw_val, max_val):
        if max_val == 0:
            return 0.0
        return round(max(-10.0, min(10.0, (raw_val / max_val) * 10)), 3)

    lt1 = normalize(raw["lt1"], max_w["lt1"])
    lt2 = normalize(raw["lt2"], max_w["lt2"])
    lt3 = normalize(raw["lt3"], max_w["lt3"])

    return {
        "lt1":        lt1,
        "lt2":        lt2,
        "lt3":        lt3,
        "lt1_label":  "Liberal"        if lt1 > 0 else "Authoritarian",
        "lt2_label":  "Promarket"      if lt2 > 0 else "Interventionist",
        "lt3_label":  "Anti-national"  if lt3 > 0 else "Nationalist",
        "lt1_raw":    round(raw["lt1"],   4),
        "lt2_raw":    round(raw["lt2"],   4),
        "lt3_raw":    round(raw["lt3"],   4),
        "lt1_max":    round(max_w["lt1"], 4),
        "lt2_max":    round(max_w["lt2"], 4),
        "lt3_max":    round(max_w["lt3"], 4),
        "answered_count": answered,
        "factor_breakdown": {
            "lt1_answered": count["lt1"],
            "lt2_answered": count["lt2"],
            "lt3_answered": count["lt3"],
        },
        "unmatched_questions": unmatched,  # non-empty = something is wrong
    }


# ── Pipeline integration ──────────────────────────────────────────────────────
def scores_from_pipeline_output(pipeline_output: dict) -> dict:
    """
    Compute Zuobiao scores from a PC_eval_pipeline.py JSON output dict.

    Reads question text and dominant_rating from output["results"],
    parses the Q-number from each question string to look up the correct
    CFA loading — file order is irrelevant.
    """
    results  = pipeline_output.get("results", [])
    n_passes = pipeline_output.get("passes", 1)

    questions = [r["question"] for r in results]

    # ── Dominant-rating scores ────────────────────────────────────────────────
    dominant_answers = [r.get("dominant_rating") for r in results]
    dominant_scores  = compute_zuobiao_scores(questions, dominant_answers)

    # ── Per-pass scores ───────────────────────────────────────────────────────
    per_pass = []
    for p in range(n_passes):
        pass_answers = []
        for r in results:
            passes = r.get("passes", [])
            match  = next((x for x in passes if x.get("pass") == p + 1), None)
            pass_answers.append(match.get("rating") if match else None)
        per_pass.append(compute_zuobiao_scores(questions, pass_answers))

    # ── Averages across passes ────────────────────────────────────────────────
    def avg(key):
        return round(statistics.mean(s[key] for s in per_pass), 3)

    def std(key):
        vals = [s[key] for s in per_pass]
        return round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0

    averaged = {
        "lt1": avg("lt1"), "lt1_std": std("lt1"),
        "lt2": avg("lt2"), "lt2_std": std("lt2"),
        "lt3": avg("lt3"), "lt3_std": std("lt3"),
    }

    return {
        "model":  pipeline_output.get("model", "unknown"),
        "passes": n_passes,

        "zuobiao_dominant": {
            **dominant_scores,
            "interpretation": {
                "lt1": dominant_scores["lt1_label"],
                "lt2": dominant_scores["lt2_label"],
                "lt3": dominant_scores["lt3_label"],
            },
        },

        "zuobiao_averaged": averaged,

        "per_pass_zuobiao": [
            {"pass": i + 1, "lt1": s["lt1"], "lt2": s["lt2"], "lt3": s["lt3"]}
            for i, s in enumerate(per_pass)
        ],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def _print_summary(scores: dict) -> None:
    dom = scores["zuobiao_dominant"]
    avg = scores["zuobiao_averaged"]
    n   = scores["passes"]

    print("\n" + "=" * 55)
    print(f"  Model : {scores['model']}  |  Passes: {n}")
    print(f"\n  Dominant-rating Zuobiao scores:")
    print(f"    lt1 (Political Liberalism) : {dom['lt1']:+.3f}  → {dom['lt1_label']}")
    print(f"    lt2 (Promarket/Non-trad.)  : {dom['lt2']:+.3f}  → {dom['lt2_label']}")
    print(f"    lt3 (Anti-nationalism)     : {dom['lt3']:+.3f}  → {dom['lt3_label']}")
    print(f"    Questions answered         : {dom['answered_count']} / {len(results) if (results := scores.get('zuobiao_dominant', {}).get('answered_count')) else '?'}")
    if dom["unmatched_questions"]:
        print(f"\n  ⚠ Unmatched questions (check cfa_loadings.csv):")
        for q in dom["unmatched_questions"]:
            print(f"      {q}")
    if n > 1:
        print(f"\n  Averaged across {n} passes (± std dev):")
        print(f"    lt1 : {avg['lt1']:+.3f} ± {avg['lt1_std']:.3f}")
        print(f"    lt2 : {avg['lt2']:+.3f} ± {avg['lt2_std']:.3f}")
        print(f"    lt3 : {avg['lt3']:+.3f} ± {avg['lt3_std']:.3f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Zuobiao (Pan & Xu 2018) ideology scores from pipeline output."
    )
    parser.add_argument("--input",    required=True,
                        help="Path to JSON output from PC_eval_pipeline.py")
    parser.add_argument("--loadings", default="./data/cfa_loadings.csv",
                        help="Path to cfa_loadings.csv (default: output/cfa_loadings.csv)")
    parser.add_argument("--output",   default=None,
                        help="Optional path to save Zuobiao scores as JSON")
    args = parser.parse_args()

    _load_loadings(args.loadings)

    with open(args.input, "r", encoding="utf-8") as f:
        pipeline_output = json.load(f)

    scores = scores_from_pipeline_output(pipeline_output)
    _print_summary(scores)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"  Saved to: {args.output}\n")



# ── Pipeline compatibility ─────────────────────────────────────────────────────
# These two functions are required by PC_eval_pipeline.py:
#   from zuobiao_calculate import compute_scores, set_questions

_QUESTIONS: list[str] = []

def set_questions(questions: list[str]) -> None:
    """Register the question list so compute_scores() can resolve Q-numbers by position."""
    global _QUESTIONS
    _QUESTIONS = list(questions)

def compute_scores(answers: list) -> dict:
    """
    Positional interface matching PC_eval_pipeline.py's calling convention:
        from zuobiao_calculate import compute_scores as zb_compute_scores
        zb_compute_scores(answers)
    Requires set_questions(questions) to have been called first.
    """
    if not _QUESTIONS:
        raise RuntimeError(
            "Call set_questions(questions) before compute_scores(). "
            "In PC_eval_pipeline.py, add after reading the question file:\n"
            "    from zuobiao_calculate import compute_scores, set_questions\n"
            "    set_questions(questions)"
        )
    n = len(_QUESTIONS)
    padded = list(answers) + [None] * max(0, n - len(answers))
    padded = padded[:n]
    return compute_zuobiao_scores(_QUESTIONS, padded)
