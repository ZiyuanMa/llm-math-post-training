import random
import time
import json
import re
from typing import Tuple, List, Dict, Any, Optional, Union, Set
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from vllm import SamplingParams


# ----- Configuration -----
TARGET_COUNT = 512+100

MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct' 
OUTPUT_FILENAME = 'data.json' 
COEFF_RANGE = (-30, 30)   
SOL_RANGE = (-30, 30)     
ALLOW_ZERO_COEFFS = False 
FLOAT_TOLERANCE = 1e-5    
BATCH_SIZE = 16           
MAX_TOKENS = 512         
TEMPERATURE = 0.7         
TOP_P = 0.8             
TOP_K = 20                
LOAD_IN_4BIT = True                   
GPU_MEMORY_UTILIZATION = 0.85 
N_GENERATIONS = 4

# ----- Equation Generation and Formatting -----

def generate_system() -> Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[int, int]]:
    """Generates a system of two linear equations with a unique integer solution."""
    while True:
        x = random.randint(SOL_RANGE[0], SOL_RANGE[1])
        y = random.randint(SOL_RANGE[0], SOL_RANGE[1])
        a = random.randint(COEFF_RANGE[0], COEFF_RANGE[1])
        b = random.randint(COEFF_RANGE[0], COEFF_RANGE[1])
        d = random.randint(COEFF_RANGE[0], COEFF_RANGE[1])
        e = random.randint(COEFF_RANGE[0], COEFF_RANGE[1])
        if not ALLOW_ZERO_COEFFS:
            if (a == 0 and b == 0) or (d == 0 and e == 0):
                continue
        determinant = a * e - b * d
        if determinant == 0:
            continue
        c = a * x + b * y
        f = d * x + e * y
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
    if not term_x and not term_y:
        left_side = "0"
    elif not term_x:
        left_side = term_y
    elif not term_y:
        left_side = term_x
    else:
        sign = " + " if b > 0 else " "
        left_side = f"{term_x}{sign}{term_y}"
    return f"{left_side} = {c}"


# ----- Unsloth/vLLM Interaction -----

def solve_batch_with_fast_generate(
    model: FastLanguageModel,    
    tokenizer: AutoTokenizer,   
    prompts: List[str],
    sampling_params: SamplingParams 
) -> List[List[str]]:
    """Uses Unsloth's model.fast_generate for batch inference with vLLM backend."""

    print(f"[INFO] Sending batch of {len(prompts)} prompts to Unsloth fast_generate (vLLM) with n={sampling_params.n}...")

    outputs = model.fast_generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True
    )

    print(f"[INFO] Received {len(outputs)} response groups from fast_generate.")

    all_generated_texts: List[List[str]] = []
    for output in outputs:
        # Extract the text from each CompletionOutput for this prompt
        prompt_outputs = [comp.text.strip() for comp in output.outputs]
        all_generated_texts.append(prompt_outputs) # Append the list of N texts

    return all_generated_texts



# ----- XML-like Parsing -----

think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
x_sol_pattern = re.compile(r"x\s*=\s*(-?\d+(?:\.\d+)?)")
y_sol_pattern = re.compile(r"y\s*=\s*(-?\d+(?:\.\d+)?)")

def parse_llm_response_xml_like(response_text: Optional[str]) -> Tuple[Optional[str], Optional[Union[int, float]], Optional[Union[int, float]]]:
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

def verify_solution(
    llm_x: Optional[Union[int, float]],
    llm_y: Optional[Union[int, float]],
    actual_x: int,
    actual_y: int,
    tolerance: float = FLOAT_TOLERANCE
) -> bool:
    """Verifies if the LLM's solution matches the actual solution within tolerance."""
    if llm_x is None or llm_y is None: return False
    try:
        return abs(float(llm_x) - actual_x) < tolerance and abs(float(llm_y) - actual_y) < tolerance
    except (ValueError, TypeError):
        return False


if __name__ == "__main__":
    # --- Initialize Unsloth Model and Tokenizer ---
    print(f"Initializing Unsloth FastLanguageModel with model '{MODEL_NAME}'...")
    # Load model and tokenizer using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_TOKENS + 512, # Sequence length for context + generation
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=True, # Explicitly enable vLLM backend for fast_generate
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION, # Control vLLM memory usage
    )
    print("Unsloth model and tokenizer initialized successfully.")
    model.eval()


    sampling_params = SamplingParams(
        n=N_GENERATIONS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
    )
    print(f"Using SamplingParams: n={sampling_params.n}, temp={sampling_params.temperature}, top_p={sampling_params.top_p}, top_k={sampling_params.top_k}, max_tokens={sampling_params.max_tokens}")


    collected_data: List[Dict[str, Any]] = []
    generated_system_coeffs_set: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()
    total_attempts = 0
    max_loops = (TARGET_COUNT // BATCH_SIZE + 1) * 5

    print(f"\nStarting generation of {TARGET_COUNT} unique solvable linear equation systems using Unsloth fast_generate...")
    print(f"Generating {N_GENERATIONS} responses per prompt and selecting the first correct one.")
    print(f"Using model '{MODEL_NAME}'.")
    print(f"Processing up to {BATCH_SIZE} prompts per loop with vLLM backend.")
    print(f"Expecting XML-like output: <think>...</think><answer>...</answer>")
    print("-" * 30)

    start_time = time.time()
    loop_count = 0

    while len(collected_data) < TARGET_COUNT:
        loop_count += 1
        if loop_count > max_loops:
            print(f"[ERROR] Exceeded maximum loops ({max_loops}). Stopping generation.")
            break

        print(f"\n[INFO] Loop {loop_count}, Collected: {len(collected_data)}/{TARGET_COUNT}. Generating prompts...")

        batch_prompts_formatted: List[str] = []
        batch_metadata: List[Dict[str, Any]] = []

        attempts_this_batch = 0
        max_attempts_for_batch = BATCH_SIZE * 5
        while len(batch_prompts_formatted) < BATCH_SIZE and attempts_this_batch < max_attempts_for_batch:
            total_attempts += 1
            attempts_this_batch += 1
            system_coeffs, actual_solution = generate_system()
            eq1_coeffs, eq2_coeffs = system_coeffs
            actual_x, actual_y = actual_solution

            canonical_coeffs_tuple = tuple(sorted((eq1_coeffs, eq2_coeffs)))
            if canonical_coeffs_tuple in generated_system_coeffs_set:
                time.sleep(0.01)
                continue

            generated_system_coeffs_set.add(canonical_coeffs_tuple)

            eq1_str = format_equation(*eq1_coeffs)
            eq2_str = format_equation(*eq2_coeffs)
            messages = [
                {"role": "system", "content": """You are a helpful assistant tasked with solving mathematical problems.
Carefully analyze the problem provided by the user and provide a step-by-step solution.
Your response MUST contain exactly one <think>...</think> block and exactly one <answer>...</answer> block.
Do NOT include any text or explanation before the <think> tag or after the </answer> tag.
Inside the <think> block, provide your step-by-step reasoning and calculations as a single string.
Inside the <answer> block, provide ONLY the final numerical solution in the specified format (e.g., x = ..., y = ...)."""},
                {"role": "user", "content": f"""Solve the following system of linear equations for the variables x and y:
Equation 1: {eq1_str}
Equation 2: {eq2_str}
Respond following the structure precisely:
<think>Reasoning steps</think>
<answer>x = value, y = value</answer>"""}
            ]

            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts_formatted.append(formatted_prompt)

            batch_metadata.append({
                "eq1_coeffs": eq1_coeffs,
                "eq2_coeffs": eq2_coeffs,
                "actual_x": actual_x,
                "actual_y": actual_y,
                "prompt_messages": messages, # <-- Store the messages list here
            })

        if not batch_prompts_formatted:
            print("[WARN] Could not generate any unique systems for this loop, continuing...")
            time.sleep(0.5)
            continue

        print(f"[INFO] Generated {len(batch_prompts_formatted)} unique formatted prompts for this loop.")

        # 2. Send the batch to the Unsloth/vLLM engine using fast_generate
        batch_responses_list = solve_batch_with_fast_generate(
            model,
            tokenizer,
            batch_prompts_formatted,
            sampling_params # Pass the vLLM sampling params
        )

        # 3. Process the batch responses (list of lists)
        successful_in_batch = 0
        # Iterate through each prompt's results
        for i, prompt_responses in enumerate(batch_responses_list):
            if len(collected_data) >= TARGET_COUNT:
                 break # Stop if target reached
            if i >= len(batch_metadata): # Safety check
                print(f"[WARN] Response index {i} out of bounds for metadata. Skipping.")
                continue

            metadata = batch_metadata[i]
            actual_x = metadata["actual_x"]
            actual_y = metadata["actual_y"]
            found_correct = False

            # Iterate through the N_GENERATIONS responses for this specific prompt
            for response_idx, response_raw in enumerate(prompt_responses):
                if response_raw is None:
                    continue # Skip if a specific response is None

                # Parse the individual response
                thinking_text, llm_x, llm_y = parse_llm_response_xml_like(response_raw)

                # Verify using the parsed llm_x and llm_y
                if verify_solution(llm_x, llm_y, actual_x, actual_y):
                    successful_in_batch += 1

                    collected_data.append({
                        "prompt": metadata["prompt_messages"], 
                        "response": response_raw,    
                        "answer": {"x": actual_x, "y": actual_y}, 
                    })
                    found_correct = True
                    break



        print(f"[INFO] Processed loop. {successful_in_batch}/{len(batch_prompts_formatted)} prompts yielded a correct answer within {N_GENERATIONS} attempts.")
        print("-" * 20)

    end_time = time.time()
    duration = end_time - start_time
    duration_minutes = duration / 60

    print("\n" + "=" * 30)
    print(f"Finished.")
    print(f"Collected {len(collected_data)} valid data points.")
    print(f"Total loops: {loop_count}")
    print(f"Total generation attempts (individual systems): {total_attempts}")
    print(f"Total generation time: {duration:.2f} seconds ({duration_minutes:.2f} minutes)")

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=4)