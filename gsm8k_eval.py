import re
import time
import math
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from typing import Union, Optional, List, Tuple

LORA = True
# --- Helper Functions ---

def parse_llm_response_gsm8k(response: str) -> Tuple[Optional[str], Optional[Union[int, float]]]:
    """
    Args:
        response: The full text response from the LLM.
    Returns:
        A tuple containing:
        - thinking_text: The text within the <think> tag (or None).
        - answer_value: The numerical value extracted from the <answer> tag (or None).
    """
    thinking_text: Optional[str] = None
    answer_value: Optional[Union[int, float]] = None

    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking_text = think_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        num_match = re.search(r"^([-+]?[\d,.-]+)$", answer_content)
        if num_match:
            num_str = num_match.group(1).replace(",", "").rstrip('.')
            try:
                answer_value = float(num_str)
                if answer_value.is_integer():
                    answer_value = int(answer_value)
            except ValueError:
                answer_value = None

    return thinking_text, answer_value

def compare_gsm8k_solutions(extracted_ans: Optional[Union[int, float]],
                            ground_truth_ans: Optional[Union[int, float]],
                            tolerance: float = 1e-6) -> bool:
    if extracted_ans is None or ground_truth_ans is None:
        return False
    try:
        return math.isclose(float(extracted_ans), float(ground_truth_ans),
                            rel_tol=tolerance, abs_tol=tolerance)
    except (ValueError, TypeError):
        return False

def load_and_preprocess_gsm8k(num_samples: Optional[int] = None) -> Optional[Dataset]:
    print("\nLoading and Preprocessing GSM8K dataset...")
    try:
        num_cpus = os.cpu_count() or 1
        gsm8k_test_raw = load_dataset("gsm8k", "main", split="test")

        def _preprocess(example):
            question = example['question']
            answer_text = example['answer']
            match = re.search(r"####\s*([\d,.-]+)", answer_text)
            final_answer_num = None
            if match:
                final_answer_str = match.group(1).replace(",", "").rstrip('.')
                try:
                    final_answer_num = float(final_answer_str)
                    if final_answer_num.is_integer():
                        final_answer_num = int(final_answer_num)
                except ValueError:
                    print(f"Warning: Could not convert GSM8K GT answer '{final_answer_str}' to number.")
            else:
                 print(f"Warning: Could not find '#### pattern' in GSM8K GT answer: {answer_text[:50]}...")

            return {'question': question, 'answer_gt': final_answer_num}

        processed_gsm8k = gsm8k_test_raw.map(_preprocess, num_proc=num_cpus)
        processed_gsm8k = processed_gsm8k.filter(lambda x: x['answer_gt'] is not None, num_proc=num_cpus)

        if num_samples is not None and num_samples < len(processed_gsm8k):
             gsm8k_eval_dataset = Dataset.from_dict(processed_gsm8k.shuffle(seed=42).select(range(num_samples))[:])
             print(f"Finished processing. Using {len(gsm8k_eval_dataset)} random samples for evaluation.")
        else:
             gsm8k_eval_dataset = Dataset.from_dict(processed_gsm8k[:])
             print(f"Finished processing. Using all {len(gsm8k_eval_dataset)} valid samples for evaluation.")

        return gsm8k_eval_dataset

    except Exception as e:
        print(f"Fatal Error: Failed to load or process GSM8K dataset: {e}")
        return None

def evaluate_gsm8k_model(llm: LLM, tokenizer, eval_dataset, lora_request: Optional[LoRARequest], sampling_params: SamplingParams):

    questions = eval_dataset['question']
    gt_answers = eval_dataset['answer_gt']
    n_outputs = sampling_params.n
    total_prompts = len(eval_dataset)

    system_prompt = (
        "You are a helpful assistant tasked with solving mathematical problems.\n"
        "Carefully analyze the problem provided by the user and provide a step-by-step solution.\n"
        "Your response MUST contain exactly one <think>...</think> block and exactly one <answer>...</answer> block.\n"
        "Do NOT include any text or explanation before the <think> tag or after the </answer> tag.\n"
        "Inside the <think> block, provide your step-by-step reasoning and calculations as a single string.\n"
        "Inside the <answer> block, provide ONLY the final numerical solution. Do NOT include units or any other text."
    )
    # -----------------------------------------------------

    eval_prompts_formatted = []
    for question in questions:
         messages = [
             {'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': question}
         ]
         formatted_p = tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
         eval_prompts_formatted.append(formatted_p)

    print(f"Generating {n_outputs} response(s) per prompt for {total_prompts} GSM8K samples...")
    request_outputs: List[RequestOutput] = llm.generate(
        eval_prompts_formatted,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    total_correct_outputs = 0
    total_generated_outputs = 0
    incorrect_details_list = []
    total_extraction_errors = 0
    total_ground_truth_errors = 0 # Note: This counter is based on dataset preprocessing, not generation
    valid_prompts_count = total_prompts # Number of prompts attempted

    eval_desc = f"Evaluating GSM8K (n={n_outputs}, avg accuracy)"
    print(f"Calculating average accuracy ({eval_desc})...")

    for i, req_output in enumerate(tqdm(request_outputs, desc=eval_desc)):
        gt_ans = gt_answers[i]
        prompt_text = req_output.prompt

        outputs_details_for_prompt = []
        prompt_had_error_or_incorrect = False # Track if any output for this prompt failed

        for j, gen_output in enumerate(req_output.outputs):
            total_generated_outputs += 1 # Count every generated output
            gen_text = gen_output.text
            _, llm_ans_val = parse_llm_response_gsm8k(gen_text)
            is_output_correct = False
            has_error = False
            error_type = None

            if llm_ans_val is None:
                total_extraction_errors += 1
                has_error = True
                error_type = "Extraction Error"
            else:
                is_output_correct = compare_gsm8k_solutions(llm_ans_val, gt_ans)
                if is_output_correct:
                    total_correct_outputs += 1 # Count each correct output
                else:
                    has_error = True
                    error_type = "Incorrect Solution"

            if has_error or not is_output_correct:
                 prompt_had_error_or_incorrect = True

            outputs_details_for_prompt.append({
                "out_idx": j,
                "full_text": gen_text,
                "is_correct": is_output_correct,
                "llm_ans": llm_ans_val,
                "error": error_type
            })
            # Removed the break statement - we need to evaluate all n outputs

        if prompt_had_error_or_incorrect:
             incorrect_details_list.append({
                "eval_idx": i, "prompt": prompt_text,
                "gt": gt_ans, "error": "Evaluation Failed (at least one output)",
                "outputs": outputs_details_for_prompt
             })

    accuracy = total_correct_outputs / total_generated_outputs if total_generated_outputs > 0 else 0

    # Return total correct outputs and total generated outputs for detailed reporting
    return accuracy, total_correct_outputs, total_extraction_errors, total_ground_truth_errors, incorrect_details_list, total_generated_outputs


if __name__ == "__main__":

    n_outputs = 3
    lora_adapter_path = "grpo_saved_lora"
    # lora_adapter_path = "sft_final_lora"
    lora_adapter_path = "outputs_sft/checkpoint-64"
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    num_gsm8k_samples = None # Set to a small number like 10 for quick testing, None for all

    print("--- GSM8K Evaluation Configuration ---")
    print(f"N Outputs per Prompt:    {n_outputs}")
    print(f"LoRA Adapter Path:       {'N/A' if not LORA else lora_adapter_path}") # Corrected LORA check
    print(f"Base Model:              {base_model_name}")
    print(f"GSM8K Samples:           {'All' if num_gsm8k_samples is None else num_gsm8k_samples}")
    print("------------------------------------")


    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llm = LLM(
        model=base_model_name,
        tokenizer=base_model_name,
        trust_remote_code=True,
        enable_lora=LORA,
        max_lora_rank=64,
        max_model_len=2048,
    )

    gsm8k_eval_dataset = load_and_preprocess_gsm8k(num_samples=num_gsm8k_samples)
    if gsm8k_eval_dataset is None or len(gsm8k_eval_dataset) == 0:
        print("Exiting due to GSM8K dataset loading error.")
        exit()
    total_samples_in_dataset = len(gsm8k_eval_dataset)


    lora_request = None
    if LORA:
        lora_request = LoRARequest(lora_name="gsm8k_eval_adapter", lora_int_id=1, lora_local_path=lora_adapter_path)

    sampling_params = SamplingParams(n=n_outputs, temperature=0.7, top_p=0.8, top_k=20, max_tokens=1024)


    incorrect_samples_to_show = []
    total_eval_time = 0
    accuracy = 0
    total_correct_outputs = 0
    total_ext_err = 0
    total_gt_err = 0 # From dataset loading
    total_generated_outputs = 0

    print(f"\n=== Starting GSM8K Evaluation (Average Accuracy, n={n_outputs}) ===")
    start_time = time.time()

    # Call the modified evaluation function
    accuracy, total_correct_outputs, total_ext_err, total_gt_err, incorrect_samples_to_show, total_generated_outputs = evaluate_gsm8k_model(
        llm, tokenizer, gsm8k_eval_dataset, lora_request, sampling_params
    )
    duration = time.time() - start_time
    total_eval_time = duration

    print(f"\n--- GSM8K Evaluation Summary (Average Accuracy, n={n_outputs}) ---")
    print(f"Average Accuracy per Output:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Correct Outputs:                 {total_correct_outputs}")
    print(f"Total Generated Outputs Attempted:     {total_generated_outputs} ({total_samples_in_dataset} prompts * {n_outputs} outputs/prompt)")
    print(f"Total Extraction Errors (all outputs): {total_ext_err}")
    print(f"Prompts with >=1 Incorrect/Error Output: {len(incorrect_samples_to_show)} / {total_samples_in_dataset}")
    print(f"Total Evaluation Time:                 {total_eval_time:.2f} seconds")
    print("="*55)

    if incorrect_samples_to_show:
        print(f"\n--- Example Prompts with Incorrect/Error Outputs (GSM8K) ---")
        num_to_show = min(5, len(incorrect_samples_to_show))
        for k in range(num_to_show):
            sample = incorrect_samples_to_show[k]
            print(f"\n----- Sample Eval Index: {sample['eval_idx']} -----")
            print(f"\n[Input Prompt]:\n{sample['prompt']}") # Be careful printing full prompts if sensitive
            print(f"\n[Ground Truth Answer]: {sample['gt']}")
            if sample.get('outputs'):
                 print(f"\n[Generated Outputs ({len(sample['outputs'])} total)]:")
                 for detail in sample['outputs']:
                     corr_str = "Correct" if detail['is_correct'] else f"Incorrect/Error ({detail.get('error', 'N/A')})"
                     print(f"\n--- Output {detail['out_idx']} ({corr_str}, Parsed: {detail.get('llm_ans')}) ---")
                     print(detail['full_text']) # Be careful printing full text if sensitive/long
            # elif sample['error'] == 'Ground Truth Error': # This case shouldn't happen post-filtering
            #      print("\n[Generated Outputs]: Not generated due to Ground Truth Error.")
            else:
                 print("\n[Generated Outputs]: Output details might be missing or structure changed.")
            print("-" * 25)

        if len(incorrect_samples_to_show) > num_to_show:
            print(f"\n... ({len(incorrect_samples_to_show) - num_to_show} more prompts with errors not shown)")
        print("-" * 55)
    else:
         # This means every single generated output across all prompts was correct AND parsable.
         print(f"\n--- All {total_generated_outputs} generated outputs were correct and successfully parsed! ---")

    print("\nEvaluation script finished.")

