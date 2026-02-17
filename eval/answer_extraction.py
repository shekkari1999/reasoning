"""
Structured Answer Extraction
- Parse model outputs to extract final answers
- Handle multiple answer formats (#### , boxed, "the answer is")
- Robust parsing with fallback strategies
"""

import re


def extract_answer(response):
    """
    Extract the final answer from a model response.

    Supports multiple formats (checked in priority order):
      1. "#### 42"          (GSM8K style)
      2. "\\boxed{42}"      (MATH/LaTeX style)
      3. "The answer is 42" (simple format)
      4. Last number in response (fallback)

    Returns:
        Extracted answer as string, or None if no answer found.
    """
    if not response:
        return None

    response = str(response)

    # format 1: #### delimiter (GSM8K style)
    match = re.search(r"####\s*(.+?)(?:\n|$)", response)
    if match:
        answer = match.group(1).strip()
        # clean up: remove commas, dollar signs, trailing periods
        answer = re.sub(r"[$,]", "", answer).rstrip(".")
        return answer

    # format 2: \boxed{...} (MATH style)
    match = re.search(r"\\boxed\{(.+?)\}", response)
    if match:
        return match.group(1).strip()

    # format 3: "The answer is ..."
    match = re.search(r"[Tt]he answer is\s*[:\s]*(.+?)(?:\.|,|\n|$)", response)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r"[$,]", "", answer).rstrip(".")
        return answer

    # format 4: last number in the response (fallback)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    if numbers:
        return numbers[-1]

    return None


def extract_all_numbers(text):
    """Extract all numbers from text."""
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def extract_gsm8k_answer(solution_text):
    """
    Extract answer from GSM8K solution format.
    GSM8K solutions end with #### <answer>
    """
    match = re.search(r"####\s*(.+?)$", solution_text, re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        answer = answer.replace(",", "")
        return answer
    return None
