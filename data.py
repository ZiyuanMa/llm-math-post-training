import random
import time
import json
import ollama
from typing import Tuple, List, Dict, Any, Optional, Union, Set

# ----- Configuration -----
TARGET_COUNT = 1024 + 50  # Target number of successful data points
N_TRIES_PER_SYSTEM = 1    # Number of times to ask the LLM for a solution for each system
OLLAMA_MODEL = 'gemma3:12b' # The Ollama model to use
OUTPUT_FILENAME = 'data_simplified.json' # Output file name
COEFF_RANGE = (-30, 30)   # Range for equation coefficients (a, b, d, e)
SOL_RANGE = (-30, 30)     # Range for the integer solutions (x, y)
ALLOW_ZERO_COEFFS = False # Whether to allow coefficients like a=0 or b=0
FLOAT_TOLERANCE = 1e-5    # Tolerance for verifying floating point solutions from LLM

# ----- Equation Generation and Formatting -----

def generate_system() -> Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[int, int]]:
    """Generates a system of two linear equations with a unique integer solution."""
    while True:
        # Generate potential integer solutions
        x = random.randint(SOL_RANGE[0], SOL_RANGE[1])
        y = random.randint(SOL_RANGE[0], SOL_RANGE[1])

        # Generate coefficients for the equations
        a = random.randint(COEFF_RANGE[0], COEFF_RANGE[1])
        b = random.randint(COEFF_RANGE[0], COEFF_RANGE[1])
        d = random.randint(COEFF_RANGE[0], COEFF_RANGE[1])
        e = random.randint(COEFF_RANGE[0], COEFF_RANGE[1])

        # Ensure coefficients are not both zero in an equation if disallowed
        if not ALLOW_ZERO_COEFFS:
            if (a == 0 and b == 0) or (d == 0 and e == 0):
                continue # Regenerate if coefficients are invalid

        # Check determinant to ensure a unique solution exists (determinant != 0)
        determinant = a * e - b * d
        if determinant == 0:
            continue # Regenerate if determinant is zero (no unique solution)

        # Calculate the constant terms (c, f) based on the solution (x, y)
        c = a * x + b * y
        f = d * x + e * y

        # Return the coefficients and the actual solution
        # Format: ((a, b, c), (d, e, f)), (x, y)
        return ((a, b, c), (d, e, f)), (x, y)

def format_term(coeff: int, var: str) -> str:
    """Formats a single term (e.g., '3x', '-y', 'x')."""
    if coeff == 0:
        return ""
    if abs(coeff) == 1:
        return f"{'-' if coeff == -1 else ''}{var}"
    return f"{coeff}{var}"

def format_equation(a: int, b: int, c: int) -> str:
    """Formats coefficients (a, b, c) into a string 'ax + by = c'."""
    term_x = format_term(a, 'x')
    term_y = format_term(b, 'y')

    # Build the left side of the equation
    if not term_x and not term_y:
        left_side = "0"
    elif not term_x:
        left_side = term_y
    elif not term_y:
        left_side = term_x
    else:
        # Add '+' sign only if the second term coefficient (b) is positive
        sign = " + " if b > 0 else " "
        # format_term handles the '-' sign for negative b
        left_side = f"{term_x}{sign}{term_y}"

    return f"{left_side} = {c}"

# ----- Prompt Formatting -----

def format_system_prompt_json(eq1_coeffs: Tuple[int, int, int], eq2_coeffs: Tuple[int, int, int]) -> str:
    """Creates the JSON-formatted prompt for the LLM."""
    eq1_str = format_equation(*eq1_coeffs)
    eq2_str = format_equation(*eq2_coeffs)
    # Example structure for the LLM's JSON output
    json_structure_example = """
{
  "reasoning": "A string containing the step-by-step derivation of the solution.",
  "solution": {
    "x": <numerical_value_for_x>,
    "y": <numerical_value_for_y>
  }
}
"""
    # The prompt instructing the LLM
    prompt = f"""Solve the following system of linear equations for the variables x and y:
Equation 1: {eq1_str}
Equation 2: {eq2_str}

Your response MUST be a single, valid JSON object.
Do NOT include any text or explanation before or after the JSON object.
Do NOT use markdown formatting like ```json ... ```.
The JSON object MUST conform exactly to the following structure:
{json_structure_example}
Replace <numerical_value_for_x> and <numerical_value_for_y> with the calculated numerical solutions for x and y, respectively. Ensure the "reasoning" field contains your step-by-step working as a single string.
"""
    return prompt

# ----- Ollama Interaction -----

def solve_with_ollama(prompt: str, model_name: str) -> Optional[str]:
    """Sends the prompt to the Ollama model and returns the response content."""

    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        format='json', # Request JSON output directly from Ollama
        options={
            'num_predict': 2048, # Max tokens for the response
            'temperature': 1.0,  # Controls randomness (lower is more deterministic)
            'top_k': 64,
            'top_p': 0.95,
        }
    )
    # Extract the content string from the response
    content = response.get('message', {}).get('content')

    return content
        

# ----- JSON Parsing -----

def parse_llm_response_json(response_text: Optional[str]) -> Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]:
    """Parses the LLM's JSON response string to extract the solution (x, y)."""
    if not response_text:
        return None, None

    try:
        data = json.loads(response_text)
        if not isinstance(data, dict):
            # Use print instead of logging.warning
            print(f"[WARN] Parsing failed: Response is not a JSON object. Received: {response_text[:200]}...")
            return None, None

        solution = data.get('solution')

        # Check if the structure is as expected
        if not isinstance(solution, dict) or 'x' not in solution or 'y' not in solution:
            # Use print instead of logging.warning
            print(f"[WARN] Parsing failed: JSON missing 'solution' dictionary with 'x' and 'y'. Keys found: {list(data.keys())}")
            return None, None

        x_val = solution['x']
        y_val = solution['y']

        # Validate the types of the solution values
        if not isinstance(x_val, (int, float)) or not isinstance(y_val, (int, float)):
            # Use print instead of logging.warning
            print(f"[WARN] Parsing failed: Solution values are not numeric. x: {x_val}, y: {y_val}")
            return None, None

        # Convert floats that are whole numbers to integers for cleaner comparison/storage
        if isinstance(x_val, float) and x_val.is_integer():
            x_val = int(x_val)
        if isinstance(y_val, float) and y_val.is_integer():
            y_val = int(y_val)

        # We only need x and y for verification now
        return x_val, y_val

    except json.JSONDecodeError as e:
        # Use print instead of logging.warning
        print(f"[WARN] Parsing failed: Invalid JSON received despite format='json'. Error: {e}. Received: {response_text[:200]}...")
        return None, None
    except Exception as e:
        # Use print instead of logging.error
        print(f"[ERROR] Parsing failed: Unexpected error during JSON parsing/validation: {e}")
        # import traceback
        # traceback.print_exc()
        return None, None

# ----- Verification -----

def verify_solution(
    llm_x: Optional[Union[int, float]],
    llm_y: Optional[Union[int, float]],
    actual_x: int, # The actual solution is always an integer in our generation
    actual_y: int,
    tolerance: float = FLOAT_TOLERANCE
) -> bool:
    """Verifies if the LLM's solution matches the actual solution within tolerance."""
    if llm_x is None or llm_y is None:
        return False
    # Compare using floating point numbers and tolerance
    try:
        return abs(float(llm_x) - actual_x) < tolerance and abs(float(llm_y) - actual_y) < tolerance
    except (ValueError, TypeError):
        # Use print instead of logging.warning
        print(f"[WARN] Verification failed: LLM solution values are not valid numbers. x: {llm_x}, y: {llm_y}")
        return False

# ----- Main Program Logic -----

def attempt_single_system(
    model_name: str,
    n_tries: int,
    generated_coeffs_set: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
) -> Optional[Dict[str, Any]]:
    """
    Generates a unique system, prompts the LLM, verifies the solution,
    and returns the required data if successful.
    Returns None if the system is a duplicate or the LLM fails.
    """
    system_coeffs, actual_solution = generate_system()
    eq1_coeffs, eq2_coeffs = system_coeffs
    actual_x, actual_y = actual_solution

    # Create a canonical (consistent) representation for checking uniqueness
    # Sort the coefficient tuples to handle cases where eq1 and eq2 are swapped
    canonical_coeffs_tuple = tuple(sorted((eq1_coeffs, eq2_coeffs)))

    # Check if this exact system (regardless of equation order) was already generated
    if canonical_coeffs_tuple in generated_coeffs_set:
        # Use print instead of logging.debug
        # print(f"[DEBUG] Duplicate system generated (coeffs: {canonical_coeffs_tuple}). Skipping.")
        time.sleep(0.05) # Small pause if duplicates are frequent
        return None # Indicate a duplicate

    # If it's new, add it to the set to prevent future duplicates
    generated_coeffs_set.add(canonical_coeffs_tuple)

    # Format equations for printing and the prompt
    eq1_str = format_equation(*eq1_coeffs)
    eq2_str = format_equation(*eq2_coeffs)
    # Use print instead of logging.info
    print(f"[INFO] Generated Unique System:")
    print(f"  Eq1: {eq1_str}")
    print(f"  Eq2: {eq2_str}")
    print(f"  Actual Solution: x = {actual_x}, y = {actual_y}")

    # Create the prompt for the LLM
    prompt = format_system_prompt_json(eq1_coeffs, eq2_coeffs)

    successful_response = None
    llm_solved_it = False

    # Try asking the LLM up to n_tries times
    for try_num in range(1, n_tries + 1):
        llm_response_raw = solve_with_ollama(prompt, model_name)

        assert llm_response_raw, "LLM response is None or empty"
 
        llm_x, llm_y = parse_llm_response_json(llm_response_raw)

        # Verify the parsed solution against the actual solution
        if verify_solution(llm_x, llm_y, actual_x, actual_y):
            # Use print instead of logging.info
            print(f"[INFO]     VERIFIED! LLM solved it on try #{try_num}.")
            successful_response = llm_response_raw # Store the successful response
            llm_solved_it = True
            break # Exit the retry loop on success
        else:
            # Use print instead of logging.warning
            print(f"[WARN]     Verification failed or JSON parsing failed on try #{try_num}.")

    # If the LLM successfully solved it within the tries
    if llm_solved_it:
        # Return only the required data
        return {
            "prompt": prompt,
            "response": successful_response,
            "answer": {"x": actual_x, "y": actual_y},
        }
    else:
        # Use print instead of logging.warning
        print(f"[WARN] LLM failed to solve this unique system within {n_tries} tries. Discarding.")
        return None # Indicate LLM failure for this system

def save_results(data: List[Dict[str, Any]], filename: str):
    """Saves the collected data list to a JSON file."""
    if not filename:
        print("[WARN] Output filename is not set. Skipping saving.")
        return
    if not data:
        print("[WARN] No systems were collected. Output file not created.")
        return

    try:
        # Add an 'id' field to each entry before saving
        for i, item in enumerate(data):
            item['id'] = i + 1

        with open(filename, 'w', encoding='utf-8') as f:
            # Dump the list of dictionaries to the JSON file
            # No special converters needed as data is simple JSON types
            json.dump(data, f, ensure_ascii=False, indent=4)
        # Use print instead of logging.info
        print(f"[INFO] Results saved to: {filename}")
    except IOError as e:
        # Use print instead of logging.error
        print(f"[ERROR] Failed to write output file {filename}: {e}")
    except TypeError as e:
        # Use print instead of logging.error
        print(f"[ERROR] Failed to serialize results to JSON: {e}")

# ----- Main Execution -----
if __name__ == "__main__":
    collected_data: List[Dict[str, Any]] = []
    # Set to keep track of unique systems generated (using coefficient tuples)
    generated_system_coeffs_set: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()
    total_attempts = 0
    # Set a reasonable upper limit on attempts to prevent infinite loops
    max_attempts = TARGET_COUNT * max(N_TRIES_PER_SYSTEM, 1) * 10 # e.g., 10x attempts factor

    # Use print instead of logging.info
    print(f"Starting generation of {TARGET_COUNT} unique solvable linear equation systems...")
    print(f"Using Ollama model '{OLLAMA_MODEL}'.")
    print(f"Max {N_TRIES_PER_SYSTEM} tries per system.")
    print("-" * 30)

    start_time = time.time()

    # Loop until the target number of data points is collected
    while len(collected_data) < TARGET_COUNT:
        total_attempts += 1
        # Safety break if too many attempts are made
        if total_attempts > max_attempts:
            print(f"[ERROR] Exceeded maximum attempts ({max_attempts}). Stopping generation.")
            break
        # Optional progress update
        if total_attempts % 50 == 0: # Print progress every 50 attempts
             print(f"[INFO] Attempt {total_attempts}, Collected: {len(collected_data)}/{TARGET_COUNT}")

        # Try to generate, solve, and verify one system
        result_data = attempt_single_system(
            OLLAMA_MODEL,
            N_TRIES_PER_SYSTEM,
            generated_system_coeffs_set
        )

        # If attempt_single_system returned data (i.e., was successful and unique)
        if result_data:
            collected_data.append(result_data)
            # Use print instead of logging.info
            print(f"[INFO] System data collected. Total collected: {len(collected_data)}/{TARGET_COUNT}")
            print("-" * 20)
            # time.sleep(0.1) # Optional short delay

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 30)
    print(f"Finished.")

    # Save the collected data to the specified file
    save_results(collected_data, OUTPUT_FILENAME)
