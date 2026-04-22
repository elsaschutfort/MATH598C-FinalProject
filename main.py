import json
import os
import torch
from utils import load_model, generate_response

# Research Configuration
# Comparing models from different ideological backgrounds:
# 1. TinyLlama: US-centric, trained on western-dominated datasets.
# 2. Qwen: Developed by Alibaba (China), subject to different cultural/legal alignment norms.
MODELS_TO_TEST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct"
    #"path/to/religious-finetuned-model",       # Religious/Secular specific
    #"path/to/reddit-finetuned-model",          # Reddit/Community specific
    #"path/to/legal-domain-model"               # Legal domain-specific
]

#Can change these that model will run
DEFAULT_BENCHMARK = "benchmark.json" #or benchmark_large.json for large file (took ~1 hr w/ Llama on GPU, Qwen on CPU)
RESULTS_FILE = "results.json" #or results_large.json
TEMPERATURE = 0.7

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
                    prompt_text = variant["text"]
                    v_type = variant["type"]
                    
                    # Ensure Likert type is identified (defaulting to likert for this project)
                    current_q_type = "likert"
                    
                    print(f"Generating: {model_id} | {q_id} [{v_type}] (Temp: {temperature})")
                    
                    response = generate_response(model, tokenizer, prompt_text, temperature=temperature)
                    
                    all_results.append({
                        "model": model_id,
                        "question_id": q_id,
                        "domain": domain,
                        "type": current_q_type,
                        "variant_type": v_type,
                        "prompt": prompt_text,
                        "response": response,
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
