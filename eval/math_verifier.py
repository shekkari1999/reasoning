"""
Math Verifier
- SymPy symbolic equivalence checking
- Handles different representations of the same answer
- Numeric fallback for non-symbolic expressions
"""

import re
from sympy import sympify, simplify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def normalize_answer(answer_str):
    """Clean and normalize answer string for comparison."""
    answer_str = str(answer_str).strip()
    # remove trailing period, dollar signs, commas in numbers, percent signs
    answer_str = answer_str.rstrip(".")
    answer_str = answer_str.replace("$", "").replace(",", "").replace("%", "")
    # remove leading/trailing whitespace
    answer_str = answer_str.strip()
    return answer_str


def string_equal(pred, target):
    """Check exact string match after normalization."""
    return normalize_answer(pred) == normalize_answer(target)


def numeric_equal(pred, target, tolerance=1e-6):
    """Check numeric equivalence within tolerance."""
    try:
        pred_val = float(normalize_answer(pred))
        target_val = float(normalize_answer(target))
        if target_val == 0:
            return abs(pred_val) < tolerance
        return abs(pred_val - target_val) / max(abs(target_val), 1e-10) < tolerance
    except (ValueError, TypeError):
        return False


def symbolic_equal(pred, target):
    """Check if two math expressions are symbolically equivalent using SymPy."""
    try:
        pred_norm = normalize_answer(pred)
        target_norm = normalize_answer(target)

        pred_expr = parse_expr(pred_norm, transformations=TRANSFORMATIONS)
        target_expr = parse_expr(target_norm, transformations=TRANSFORMATIONS)

        diff = simplify(pred_expr - target_expr)
        return diff == 0
    except Exception:
        return False


def verify_answer(pred, target):
    """
    Verify if predicted answer matches target.

    Tries in order:
    1. Exact string match (after normalization)
    2. Numeric equivalence (within tolerance)
    3. Symbolic equivalence (via SymPy)

    Returns True if any check passes.
    """
    if pred is None or target is None:
        return False

    pred = str(pred)
    target = str(target)

    # 1. string match
    if string_equal(pred, target):
        return True

    # 2. numeric match
    if numeric_equal(pred, target):
        return True

    # 3. symbolic match
    if symbolic_equal(pred, target):
        return True

    return False
