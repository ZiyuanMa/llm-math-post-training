import json
import re
import time
import math
import os
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Union, Optional, List, Tuple

# --- Helper Functions ---

def parse_llm_response_linear_equation(response: str) -> Tuple[Optional[str], Optional[Union[int, float]], Optional[Union[int, float]]]:
    thinking_text: Optional[str] = None
    x_value: Optional[Union[int, float]] = None
    y_value: Optional[Union[int, float]] = None
    solution_tag_match = re.search(r"<answer>", response, re.IGNORECASE)

    if solution_tag_match:
        thinking_text = response[:solution_tag_match.start()].strip()
        solution_part = response[solution_tag_match.end():].strip()
        solution_part = re.sub(r"</answer>\s*$", "", solution_part, flags=re.IGNORECASE).strip()
    else:
        solution_part = response.strip()
        if not re.search(r'x\s*=\s*[-+]?\d+(\.\d+)?', solution_part, re.IGNORECASE) and \
           not re.search(r'y\s*=\s*[-+]?\d+(\.\d+)?', solution_part, re.IGNORECASE):
             thinking_text = solution_part
             solution_part = ""

    x_match = re.search(r"x\s*=\s*([-+]?\d+(\.\d+)?)", solution_part, re.IGNORECASE)
    y_match = re.search(r"y\s*=\s*([-+]?\d+(\.\d+)?)", solution_part, re.IGNORECASE)
    if x_match:
        try:
            x_value = float(x_match.group(1))
        except ValueError:
            # print(f"Warning: Could not convert extracted x value '{x_match.group(1)}' to float.")
            pass
    if y_match:
        try:
            y_value = float(y_match.group(1))
        except ValueError:
            # print(f"Warning: Could not convert extracted y value '{y_match.group(1)}' to float.")
            pass

    if solution_tag_match and x_value is None and y_value is None:
         thinking_text = response.strip()
    elif not solution_tag_match and x_value is None and y_value is None and thinking_text is None:
         thinking_text = response.strip()
    return thinking_text, x_value, y_value

def compare_solutions(llm_x, llm_y, gt_x, gt_y, tolerance=1e-6) -> bool:
    if llm_x is None or llm_y is None or gt_x is None or gt_y is None: return False
    try:
        match = math.isclose(float(llm_x), float(gt_x), rel_tol=tolerance, abs_tol=tolerance) and \
                math.isclose(float(llm_y), float(gt_y), rel_tol=tolerance, abs_tol=tolerance)

        return match
    except (ValueError, TypeError):
        return False

def load_data_from_json(tokenizer, json_path="./data.json") -> Dataset:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for idx, entry in enumerate(data):
        if not (isinstance(entry, dict) and 'prompt' in entry and 'answer' in entry): continue
        prompt_list, answer_dict = entry['prompt'], entry['answer']
        if not (isinstance(prompt_list, list) and all(isinstance(p, dict) and 'role' in p and 'content' in p for p in prompt_list)): continue
        if not (isinstance(answer_dict, dict) and 'x' in answer_dict and 'y' in answer_dict): continue
        try: float(answer_dict['x']); float(answer_dict['y'])
        except (ValueError, TypeError): continue
        fmt_prompt = tokenizer.apply_chat_template(prompt_list, tokenize=False, add_generation_prompt=True)
        processed_data.append({'prompt': fmt_prompt, 'answer': answer_dict, 'original_index': idx})

    if not processed_data: print("Warning: No valid data loaded after filtering.")
    dataset = Dataset.from_dict({k: [d[k] for d in processed_data] for k in processed_data[0]}) if processed_data else \
              Dataset.from_dict({'prompt': [], 'answer': [], 'original_index': []})
    print(f"Loaded {len(dataset)} valid samples from {json_path}")
    return dataset

def evaluate_model(llm: LLM, tokenizer, eval_dataset, lora_request: LoRARequest, sampling_params, eval_mode='best_of_n'):

    eval_prompts = eval_dataset['prompt']
    gt_answers = eval_dataset['answer']
    original_indices = eval_dataset['original_index']
    n_outputs = sampling_params.n
    total_prompts = len(eval_dataset)

    print(f"Generating {n_outputs} response(s) per prompt for {total_prompts} samples...")
    request_outputs: List[RequestOutput] = llm.generate(
        eval_prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    prompt_correct_count = 0
    total_correct_outputs = 0
    incorrect_details_list = []
    total_extraction_errors = 0
    total_ground_truth_errors = 0
    valid_prompts_count = 0

    eval_desc = f"Evaluating ({eval_mode})"
    print(f"Calculating accuracy ({eval_mode})...")

    for i, req_output in enumerate(tqdm(request_outputs, desc=eval_desc)):
        gt_dict = gt_answers[i]
        original_idx = original_indices[i]
        prompt_text = req_output.prompt

        gt_x, gt_y = None, None
        try:
            gt_x_val, gt_y_val = gt_dict.get('x'), gt_dict.get('y')
            if gt_x_val is None or gt_y_val is None: raise ValueError("Missing GT")
            gt_x, gt_y = float(gt_x_val), float(gt_y_val)

        except (ValueError, TypeError, AttributeError) as e:
            total_ground_truth_errors += 1
            print(f"DEBUG: Prompt {i} (Orig: {original_idx}) - GT Error: {e}, GT Dict: {gt_dict}") # DEBUG GT Error
            incorrect_details_list.append({
                "eval_idx": i, "orig_idx": original_idx, "prompt": prompt_text,
                "gt": gt_dict, "error": "Ground Truth Error", "outputs": []
            })
            continue

        valid_prompts_count += 1
        prompt_at_least_one_correct = False
        outputs_details_for_prompt = []
        prompt_had_error_or_incorrect = False

        for j, gen_output in enumerate(req_output.outputs):
            gen_text = gen_output.text
            _, llm_x, llm_y = parse_llm_response_linear_equation(gen_text)
            is_output_correct = False
            has_error = False
            error_type = None # Initialize error_type

            if llm_x is None or llm_y is None:
                total_extraction_errors += 1
                has_error = True
                error_type = "Extraction Error"
            else:
                is_output_correct = compare_solutions(llm_x, llm_y, gt_x, gt_y)

                if is_output_correct:
                    total_correct_outputs += 1
                    prompt_at_least_one_correct = True
                    # print(f"    DEBUG: CORRECT! total_correct_outputs = {total_correct_outputs}") # DEBUG Correct
                else:
                    has_error = True
                    error_type = "Incorrect Solution"
                    # print(f"    DEBUG: Incorrect Solution.") # DEBUG Incorrect

            if has_error or not is_output_correct:
                 prompt_had_error_or_incorrect = True

            outputs_details_for_prompt.append({
                "out_idx": j,
                "full_text": gen_text,
                "is_correct": is_output_correct,
                # Add parsed values for easier debugging in final report
                "llm_x": llm_x,
                "llm_y": llm_y,
                "error": error_type
            })

            if (eval_mode == 'best_of_n' or eval_mode == 'standard') and prompt_at_least_one_correct:
                break

        should_record_incorrect = False
        if eval_mode == 'standard' or eval_mode == 'best_of_n':
            if not prompt_at_least_one_correct:
                should_record_incorrect = True
        elif eval_mode == 'average_per_output':
             if prompt_had_error_or_incorrect:
                 should_record_incorrect = True

        if should_record_incorrect:
             incorrect_details_list.append({
                "eval_idx": i, "orig_idx": original_idx, "prompt": prompt_text,
                "gt": gt_dict, "error": "Evaluation Failed",
                "outputs": outputs_details_for_prompt
             })

        if (eval_mode == 'standard' or eval_mode == 'best_of_n') and prompt_at_least_one_correct:
            prompt_correct_count += 1

    accuracy = 0
    result_count = 0
    denominator = 1

    if eval_mode == 'standard' or eval_mode == 'best_of_n':
        denominator = valid_prompts_count
        accuracy = prompt_correct_count / denominator if denominator > 0 else 0
        result_count = prompt_correct_count
    elif eval_mode == 'average_per_output':
        denominator = valid_prompts_count * n_outputs
        accuracy = total_correct_outputs / denominator if denominator > 0 else 0
        result_count = total_correct_outputs

    # --- DEBUG ---
    # print(f"\nDEBUG FINAL COUNTS: prompt_correct_count={prompt_correct_count}, total_correct_outputs={total_correct_outputs}")
    # print(f"DEBUG FINAL DENOMINATOR: {denominator} (valid_prompts={valid_prompts_count})")
    # print(f"DEBUG FINAL ACCURACY: {accuracy}")
    # -----------

    return accuracy, result_count, total_extraction_errors, total_ground_truth_errors, incorrect_details_list, denominator


# --- Main Execution Block ---
if __name__ == "__main__":


    EVALUATION_METHOD = 'average'  # 'average', 'best_of_n'

    n_outputs = 5
    dataset_path = "./data.json"
    lora_adapter_path = "grpo_saved_lora" 
    # lora_adapter_path = "sft_final_lora" 
    # lora_adapter_path = "outputs_sft/checkpoint-80"
    lora_adapter_path = "outputs_grpo/checkpoint-48"
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"


    print("--- Evaluation Configuration ---")
    print(f"Selected Method:         {EVALUATION_METHOD}")
    if EVALUATION_METHOD == 'average': print(f"  N Outputs per Prompt:  {n_outputs} (for avg per output)")
    elif EVALUATION_METHOD == 'best_of_n': print(f"  N for Best-of-N:       {n_outputs}")
    print(f"Dataset Path:            {dataset_path}")
    print(f"LoRA Adapter Path:       {lora_adapter_path}")
    print(f"Base Model:              {base_model_name}")
    print("-----------------------------")

    print("\nLoading common components...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llm = LLM(
        model=base_model_name,
        tokenizer=base_model_name,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
    )

    full_dataset = load_data_from_json(tokenizer, dataset_path)
    if len(full_dataset) < 100: raise ValueError("Dataset too small.")
    eval_dataset = full_dataset.select(range(len(full_dataset) - 100, len(full_dataset)))
    total_samples = len(eval_dataset)
    print(f"Using last {total_samples} samples for evaluation.")

    try:
        if not os.path.isdir(lora_adapter_path):
             raise FileNotFoundError(f"LoRA directory not found: {lora_adapter_path}")
        config_path = os.path.join(lora_adapter_path, "adapter_config.json")
        model_path_bin = os.path.join(lora_adapter_path, "adapter_model.bin")
        model_path_safe = os.path.join(lora_adapter_path, "adapter_model.safetensors")
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"adapter_config.json not found in {lora_adapter_path}")
        if not os.path.exists(model_path_bin) and not os.path.exists(model_path_safe):
             raise FileNotFoundError(f"adapter_model.* not found in {lora_adapter_path}")
        lora_request = LoRARequest(lora_name="grpo_adapter", lora_int_id=1, lora_local_path=lora_adapter_path)
        print(f"LoRA request created for path: {lora_adapter_path}")
    except Exception as e:
        print(f"Error preparing LoRA request for path '{lora_adapter_path}': {e}")
        exit()

    print("Common components loaded.")


    sampling_params_multi = SamplingParams(n=n_outputs, temperature=0.7, top_p=0.8, top_k=20, max_tokens=512)

    incorrect_samples_to_show = []
    total_eval_time = 0
    accuracy = 0
    correct_count_or_outputs = 0
    total_ext_err = 0
    total_gt_err = 0
    denominator = 1

    start_time = time.time()
    if EVALUATION_METHOD == 'average':
        print(f"\n=== Starting Method: Average Accuracy per Output (N={n_outputs}) ===")
        accuracy, correct_count_or_outputs, total_ext_err, total_gt_err, incorrect_samples_to_show, denominator = evaluate_model(
            llm, tokenizer, eval_dataset, lora_request, sampling_params_multi, eval_mode='average_per_output'
        )
        duration = time.time() - start_time
        total_eval_time = duration

        print(f"\n--- Average Per Output Method Summary (N={n_outputs}) ---")
        print(f"Average Accuracy per Generated Output: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total Correct Individual Outputs:      {correct_count_or_outputs} / {denominator if denominator > 0 else 'N/A'}")

    elif EVALUATION_METHOD == 'best_of_n':
        print(f"\n=== Starting Method: Best-of-N (N={n_outputs}) ===")
        accuracy, correct_count_or_outputs, total_ext_err, total_gt_err, incorrect_samples_to_show, denominator = evaluate_model(
            llm, tokenizer, eval_dataset, lora_request, sampling_params_multi, eval_mode='best_of_n'
        )
        duration = time.time() - start_time
        total_eval_time = duration

        print(f"\n--- Best-of-N Method Summary (N={n_outputs}) ---")
        print(f"Accuracy (>=1 of {n_outputs} correct): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Prompts with >=1 correct output:        {correct_count_or_outputs} / {denominator if denominator > 0 else 'N/A'}")
        print(f"Prompts where all {n_outputs} outputs failed: {denominator - correct_count_or_outputs if denominator > 0 else 'N/A'}")

    else:
        print(f"Error: Invalid EVALUATION_METHOD: '{EVALUATION_METHOD}'. Choose 'average' or 'best_of_n'.")
        exit()


    if EVALUATION_METHOD in ['average', 'best_of_n']:
        print(f"Total Extraction Errors across all outputs: {total_ext_err}")
        print(f"Total Ground Truth Errors (skipped prompts): {total_gt_err}")
        print(f"Total Evaluation Time:                  {total_eval_time:.2f} seconds")
        print("="*55)
