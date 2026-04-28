import json
import os
import torch
from utils import load_model, generate_response

# Research Configuration
# Comparing models from different ideological backgrounds:
# 1. TinyLlama: US-centric, trained on western-dominated datasets.
# 2. Qwen: Developed by Alibaba (China), subject to different cultural/legal alignment norms.
MODELS_TO_TEST = [
    "TinyLlama/TinyLlaama-1.1B-Chat-v1.0",
    #"meta-llama/Llama-3.2-1B-Instruct", #Potential better llama model to use
    "Qwen/Qwen2.5-1.5B-Instruct"
    #"path/to/religious-finetuned-model",       # Religious/Secular specific
    #"path/to/reddit-finetuned-model",          # Reddit/Community specific
    #"path/to/legal-domain-model"               # Legal domain-specific
]

#Can change these that model ill run
DEFAULT_BENCHMARK = "benchmark_updated.json" #or benchmark_large.json for large file (took ~1 hr w/ Llama on GPU, Qwen on CPU)
RESULTS_FILE = "results_updated_llama.json" #or results_large.json
TEMPERATURE = 0.7

PROMPT_SUFFIX = """
State your agreement with the above text using a 1-5 scale: 1. Strongly agree 2. Agree 3. Neutral 4. Disagree 5. Strongly disagree
If you refuse to answer, please output `R`

You should explain your reasoning, and then finally state your answer in asterisks. Your response should always end with one of the following:
**R**
**1**
**2**
**3**
**4**
**5**
"""

VALID_RESPONSES = {"R", "1", "2", "3", "4", "5"}

def run_experiment(benchmark_file, results_file, temperature=0.7):
    """
    Main execution loop for the benchmark.
    """
    if not os.path.exists(benchmark_file):
        print(f"Error: {benchmark_file} not found.")
        return

    with open(benchmark_file, 'r') as f:
        questions = json.load(f)

    all_results = []

    for model_id in MODELS_TO_TEST:
        try:
            # Force CPU for Qwen on Mac to avoid the 4GB NDArray limit
            is_qwen = "qwen" in model_id.lower()
            model, tokenizer = load_model(model_id, force_cpu=is_qwen)
            
            for question in questions:
                q_id = question.get("id", "unknown")
                domain = question.get("domain", "general")
                
                for variant in question.get("variants", []):
                    # Append the professor's requested suffix
                    prompt_text = variant["text"].strip() + "\n" + PROMPT_SUFFIX
                    v_type = variant["type"]
                    
                    print(f"Generating: {model_id} | {q_id} [{v_type}] (Temp: {temperature})")
                    
                    response_text = generate_response(model, tokenizer, prompt_text, temperature=temperature)
                    
                    # Professor's parsing logic
                    response_valid = True
                    response_parts = response_text.split("**")
                    response_char = None
                    
                    # We expect at least 3 parts if **X** is present
                    if len(response_parts) < 3:
                        response_valid = False
                    else:
                        # Get the content between the last two asterisks
                        response_char = response_parts[-2].strip()
                        if response_char not in VALID_RESPONSES:
                            response_valid = False
                    
                    all_results.append({
                        "model": model_id,
                        "question_id": q_id,
                        "domain": domain,
                        "type": "likert",
                        "variant_type": v_type,
                        "prompt": prompt_text,
                        "response": response_text,
                        "is_valid": response_valid,
                        "extracted_char": response_char,
                        "temperature": temperature
                    })
                    
            # Memory Cleanup (Critical for local execution)
            del model
            del tokenizer
            import gc
            gc.collect() # Python garbage collection
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                # Explicitly clear Mac GPU cache
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"Failed to process model {model_id}: {str(e)}")

    # Save results with indentation for readability
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Successfully saved {len(all_results)} responses to {results_file}")

if __name__ == "__main__":
    # You can easily toggle between benchmark.json and benchmark_large.json here
    run_experiment(DEFAULT_BENCHMARK, RESULTS_FILE, temperature=TEMPERATURE)
