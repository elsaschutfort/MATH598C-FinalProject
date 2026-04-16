import json
import os
from utils import load_model, generate_response

# Placeholder list of models based on project proposal
# Replace these with specific HuggingFace IDs relevant to your research
MODELS_TO_TEST = [
    # "meta-llama/Meta-Llama-3-8B-Instruct",      # General instruction baseline
    "microsoft/phi-2",                         # Another small general model
    "TheBloke/Llama-2-7b-Chat-GPTQ",           # Example general model
    "path/to/religious-finetuned-model",       # Religious/Secular specific
    "path/to/reddit-finetuned-model",          # Reddit/Community specific
    "path/to/legal-domain-model"               # Legal domain-specific
]

def run_experiment(benchmark_file, results_file):
    with open(benchmark_file, 'r') as f:
        questions = json.load(f)

    results = []

    for model_id in MODELS_TO_TEST:
        try:
            model, tokenizer = load_model(model_id)
            
            for question in questions:
                q_id = question["id"]
                domain = question["domain"]
                
                for variant in question["variants"]:
                    prompt_text = variant["text"]
                    v_type = variant["type"]
                    
                    print(f"Generating for {model_id} - {q_id} ({v_type})...")
                    response = generate_response(model, tokenizer, prompt_text)
                    
                    results.append({
                        "model": model_id,
                        "question_id": q_id,
                        "domain": domain,
                        "variant_type": v_type,
                        "prompt": prompt_text,
                        "response": response
                    })
                    
            # Clear memory after each model to avoid OOM
            del model
            del tokenizer
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing model {model_id}: {e}")

    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    benchmark_path = "benchmark.json"
    results_path = "results.json"
    
    if not os.path.exists(benchmark_path):
        print(f"Benchmark file {benchmark_path} not found.")
    else:
        run_experiment(benchmark_path, results_path)
