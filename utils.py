import re
from typing import Tuple, List, Dict, Any, Optional, Union, Set

think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
x_sol_pattern = re.compile(r"x\s*=\s*(-?\d+(?:\.\d+)?)")
y_sol_pattern = re.compile(r"y\s*=\s*(-?\d+(?:\.\d+)?)")

def parse_llm_response_linear_equation(response_text: Optional[str]) -> Tuple[Optional[str], Optional[Union[int, float]], Optional[Union[int, float]]]:
    """Parses the LLM's XML-like response to extract reasoning and solution (x, y)."""
    if not response_text:
        return None, None, None

    response_text = response_text.strip()
    # Find the first match
    think_match = think_pattern.search(response_text)
    answer_match = answer_pattern.search(response_text)

    thinking_text = think_match.group(1).strip() if think_match else None
    answer_text = answer_match.group(1).strip() if answer_match else None

    # Basic check
    if not thinking_text or not answer_text:
        return thinking_text, None, None

    x_match = x_sol_pattern.search(answer_text)
    y_match = y_sol_pattern.search(answer_text)
    llm_x: Optional[Union[int, float]] = None
    llm_y: Optional[Union[int, float]] = None

    try:
        if x_match:
            x_str = x_match.group(1)
            llm_x = float(x_str)
            if llm_x.is_integer(): llm_x = int(llm_x)

        if y_match:
            y_str = y_match.group(1)
            llm_y = float(y_str)
            if llm_y.is_integer(): llm_y = int(llm_y)

        if llm_x is None or llm_y is None:
             return thinking_text, None, None

        return thinking_text, llm_x, llm_y

    except ValueError as e:
        return thinking_text, None, None
    except Exception as e:
        print(f"[ERROR] Parsing failed: Unexpected error: {e}")
        return thinking_text, None, None
    

def parse_llm_response_gsm8k(response_text: Optional[str]) -> Tuple[Optional[str], Optional[Union[int, float]], Optional[Union[int, float]]]:
    """Parses the LLM's XML-like response to extract reasoning and solution."""
    if not response_text:
        return None, None, None

    response_text = response_text.strip()
    # Find the first match
    think_match = think_pattern.search(response_text)
    answer_match = answer_pattern.search(response_text)

    thinking_text = think_match.group(1).strip() if think_match else None
    answer_text = answer_match.group(1).strip() if answer_match else None

    return thinking_text, answer_text