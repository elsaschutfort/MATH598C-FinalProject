import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id):
    """
    Loads a model and tokenizer from HuggingFace.
    """
    print(f"Loading model: {model_id}...")
    # Detect common GPTQ identifiers and fail fast with actionable advice
    if "gptq" in model_id.lower() or model_id.lower().endswith("-gptq"):
        raise ImportError(
            "Detected a GPTQ-quantized model. Loading GPTQ models requires the `optimum` package. "
            "Install it with: `pip install optimum` (and optionally `pip install -U accelerate bitsandbytes`)."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Using device_map="auto" for automatic multi-GPU/CPU placement
    # Using 4-bit or 8-bit quantization if needed for large models (requires bitsandbytes)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """
    Generates a response from the model for a given prompt.
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
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()
