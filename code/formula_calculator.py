"""
Political Compass formula calculator.
Converted from politicalcompass.github.io's formula_calculator.js

Answer index mapping:
    0 = strongly disagree
    1 = disagree
    2 = agree
    3 = strongly agree

Each entry in econv/socv is [SD, D, A, SA] weights for that question.
"""

# Economic axis weights per question (62 questions, index 0-61)
econv = [
    [7, 5, 0, -2],   # q1
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [7, 5, 0, -2],   # q8
    [-7, -5, 0, 2],
    [6, 4, 0, -2],
    [7, 5, 0, -2],
    [-8, -6, 0, 2],
    [8, 6, 0, -2],
    [8, 6, 0, -1],
    [7, 5, 0, -3],
    [8, 6, 0, -1],
    [-7, -5, 0, 2],
    [-7, -5, 0, 1],
    [-6, -4, 0, 2],
    [6, 4, 0, -1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],   # q22
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-8, -6, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-10, -8, 0, 1],
    [-5, -4, 0, 1],
    [0, 0, 0, 0],   # q40
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],   # q52
    [0, 0, 0, 0],
    [-9, -8, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],   # q57
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],   # q62
]

# Social axis weights per question (62 questions, index 0-61)
socv = [
    [0, 0, 0, 0],    # q1
    [-8, -6, 0, 2],
    [7, 5, 0, -2],
    [-7, -5, 0, 2],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [7, 5, 0, -2],
    [0, 0, 0, 0],    # q8
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-6, -4, 0, 2],  # q22
    [7, 6, 0, -2],
    [-5, -4, 0, 2],
    [0, 0, 0, 0],
    [8, 4, 0, -2],
    [-7, -5, 0, 2],
    [-7, -5, 0, 3],
    [6, 4, 0, -3],
    [6, 3, 0, -2],
    [-7, -5, 0, 3],
    [-9, -7, 0, 2],
    [-8, -6, 0, 2],
    [7, 6, 0, -2],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [-7, -4, 0, 2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [7, 5, 0, -3],   # q40
    [-9, -6, 0, 2],
    [-8, -6, 0, 2],
    [-8, -6, 0, 2],
    [-6, -4, 0, 2],
    [-8, -6, 0, 2],
    [-7, -5, 0, 2],
    [-8, -6, 0, 2],
    [-5, -3, 0, 2],
    [-7, -5, 0, 2],
    [7, 5, 0, -2],
    [-6, -4, 0, 2],
    [-7, -5, 0, 2],  # q52
    [-6, -4, 0, 2],
    [0, 0, 0, 0],
    [-7, -5, 0, 2],
    [-6, -4, 0, 2],
    [-7, -6, 0, 2],  # q57
    [7, 6, 0, -2],
    [7, 5, 0, -2],
    [8, 6, 0, -2],
    [-8, -6, 0, 2],
    [-6, -4, 0, 2],  # q62
]

# Normalization constants from the JS
ECON_DIVISOR = 8.0
SOC_DIVISOR = 19.5
ECON_OFFSET = 0.38
SOC_OFFSET = 2.41

# Answer string -> index mapping
ANSWER_TO_IDX = {
    "strongly disagree": 0,
    "disagree": 1,
    "agree": 2,
    "strongly agree": 3,
}


def compute_scores(answers: list[str | None]) -> dict:
    """
    Compute Political Compass economic and social scores.

    Args:
        answers: List of 62 answer strings, each one of:
                 "strongly disagree", "disagree", "agree", "strongly agree", or None (skipped)

    Returns:
        dict with keys: economic_score, social_score, econ_sum, soc_sum, answered_count
    """
    if len(answers) != 62:
        raise ValueError(f"Expected 62 answers, got {len(answers)}")

    sum_e = 0.0
    sum_s = 0.0
    answered = 0

    for i, answer in enumerate(answers):
        if answer is None:
            continue
        idx = ANSWER_TO_IDX.get(answer.strip().lower())
        if idx is None:
            continue  # skip neutral / unrecognized
        sum_e += econv[i][idx]
        sum_s += socv[i][idx]
        answered += 1

    economic_score = round((sum_e / ECON_DIVISOR) + ECON_OFFSET, 2)
    social_score = round((sum_s / SOC_DIVISOR) + SOC_OFFSET, 2)

    # Clamp to [-10, 10] like the JS does
    economic_score = max(-10.0, min(10.0, economic_score))
    social_score = max(-10.0, min(10.0, social_score))

    return {
        "economic_score": economic_score,
        "social_score": social_score,
        "econ_sum": sum_e,
        "soc_sum": sum_s,
        "answered_count": answered,
    }


if __name__ == "__main__":
    # Quick sanity check — all neutral answers should give baseline offsets
    test = ["agree"] * 62  # agree = index 2, all econv/socv[i][2] = 0
    result = compute_scores(test)
    print(f"All-agree test (should be econ=0.38, social=2.41): {result}")