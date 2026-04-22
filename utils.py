import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id, force_cpu=False):
    """
    Loads a model and tokenizer from HuggingFace.
    Optimizes for Mac GPU (MPS) or CUDA if available, unless force_cpu is True.
    """
    print(f"Loading model: {model_id}...")
    
    # Detect common GPTQ identifiers... (keep previous logic)
    if "gptq" in model_id.lower() or model_id.lower().endswith("-gptq"):
        raise ImportError("Detected a GPTQ-quantized model...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Determine device
    if force_cpu:
        device = "cpu"
        dtype = torch.float32
        print("Forcing CPU mode for this model.")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Using Mac GPU (MPS) for acceleration.")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
        print("No GPU found. Running on CPU (slow).")

    # Use low_cpu_mem_usage to prevent memory spikes during loading
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "mps" or device == "cpu":
        model = model.to(device)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    """
    Generates a response from the model for a given prompt.
    Supports chat templates if available.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9 if temperature > 0 else 1.0
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()
