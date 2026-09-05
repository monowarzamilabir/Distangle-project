# -*- coding: utf-8 -*-
"""Answer extraction and scoring for the Bengali math eval.

Shared by 07_eval_math_v2.ipynb and the probe scripts so that one definition
governs every number reported in the paper.

Why this exists as its own module: the original extractor in 04 returned the
whole remainder of the line after "উত্তর:", so any answer with trailing words
or LaTeX markup failed to normalise and scored wrong even when the model was
right. TigerLLM's house style is "উত্তর: \\( ৯ \\)।", which failed 100% of the
time. Qwen3-4B produced "উত্তর: 20 জন ছাত্র..." for a gold answer of ২০ and was
also marked wrong. Both were correct.
"""
import math
import re

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# A signed integer or decimal in either Bengali or ASCII digits, allowing
# thousands separators.
_NUM = r"[-−]?[\d০-৯]+(?:[,\u09EC\d০-৯]*)?(?:[.][\d০-৯]+)?"

# Markup that wraps answers but carries no numeric meaning.
_STRIP = [
    r"\\+[()\[\]]",        # \( \) \[ \]
    r"\$+",                 # $ and $$
    r"\*+",                 # markdown bold
    r"\\text\{[^}]*\}",    # \text{...}
    r"[`~]",
]


def _clean(s):
    s = str(s)
    for pat in _STRIP:
        s = re.sub(pat, " ", s)
    return s


def first_number(text):
    """First numeric token in `text`, or None. Bengali digits included."""
    if text is None:
        return None
    m = re.search(_NUM, _clean(text))
    return m.group(0) if m else None


def last_number(text):
    if text is None:
        return None
    nums = re.findall(_NUM, _clean(text))
    return nums[-1] if nums else None


def extract_answer(text):
    """Pull a final answer out of a generation.

    None = no answer found, which is a FORMAT FAILURE and is tracked separately
    from a wrong answer. Each branch narrows to a region, then takes the first
    number inside it -- taking the whole region was the bug in 04.
    """
    if not text:
        return None
    t = str(text)

    m = re.search(r"<answer>(.*?)</answer>", t, re.S)
    if m:
        return first_number(m.group(1)) or m.group(1).strip()

    # Innermost braces, so \boxed{\frac{1}{2}} does not swallow the trailing }.
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", t)
    if m:
        return first_number(m.group(1)) or m.group(1).strip()

    # Bengali answer marker. Prefer the LAST one: models often restate the
    # answer at the end after working through the problem.
    matches = list(re.finditer(r"(?:উত্তর|উত্তরঃ|সঠিক উত্তর)\s*[:ঃ]?\s*(.+)", t))
    if matches:
        for m in reversed(matches):
            region = m.group(1).strip().split("\n")[0]
            n = first_number(region)
            if n is not None:
                return n
        return matches[-1].group(1).strip().split("\n")[0] or None

    return last_number(t)


def normalise_answer(ans):
    if ans is None:
        return None
    s = str(ans).translate(BN_DIGITS).strip()
    s = _clean(s)
    s = s.replace("−", "-")
    for ch in (",", "৳", "$", "%", " "):
        s = s.replace(ch, "")
    s = s.rstrip(".।")
    try:
        f = float(s)
        # A degenerate generation can emit hundreds of digits; float() then
        # returns inf and int(inf) raises OverflowError. Seen for real on
        # TigerLLM, which loops digits when it fails. Compare as a string in
        # that case -- such an answer is never going to match a gold value.
        if math.isinf(f) or math.isnan(f):
            return s.lower()
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return s.lower()


def is_correct(pred, gold):
    p, g = normalise_answer(pred), normalise_answer(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    # Tolerate float/int drift (20.0 vs 20) that string compare would miss.
    try:
        fp, fg = float(p), float(g)
        if math.isinf(fp) or math.isnan(fp) or math.isinf(fg) or math.isnan(fg):
            return False
        return abs(fp - fg) < 1e-6
    except (ValueError, TypeError, OverflowError):
        return False


def bengali_ratio(text):
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return len([c for c in letters if "\u0980" <= c <= "\u09ff"]) / len(letters)


if __name__ == "__main__":
    # Realistic completions, taken verbatim from actual model output.
    CASES = [
        ("উত্তর: 20 জন ছাত্র D এর চেয়ে কম নম্বর পেয়েছে।", "২০", True),
        ("উত্তর: \\( ৯ \\)।", "৯", True),
        ("উত্তর: ১২৬ টাকা", "১২৬", True),
        ("**উত্তর:** পূর্ণসংখ্যার এককের অঙ্ক হল 2", "২", True),
        ("উত্তর: ২০", "২০", True),
        ("অতএব উত্তর: $148$ জন", "১৪৮", True),
        ("the answer is \\boxed{42}", "42", True),
        ("<answer>7</answer>", "7", True),
        ("উত্তর: ১২৬ টাকা", "১২৭", False),
        ("blah 13 blah", "42", False),
        ("no digits at all", "5", False),
        ("উত্তর: 20", "20.0", True),
        # Regression: a 400-digit run makes float() return inf and
        # int(inf) raise OverflowError. Crashed a real eval pass.
        ("উত্তর: " + "9" * 400, "৮", False),
        ("উত্তর: " + "1" * 500, "1" * 500, True),
    ]
    bad = 0
    for text, gold, want in CASES:
        got = is_correct(extract_answer(text), gold)
        flag = "ok  " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"{flag} {text[:48]!r:<52} gold={gold!r:<8} -> {got}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} passed")
    raise SystemExit(1 if bad else 0)
