import json
from collections import OrderedDict

def transform_results(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    VALID_RESPONSES = {"R", "1", "2", "3", "4", "5"}
    transformed = []

    for item in data:
        response_text = item.get("response", "")
        
        # Professor's parsing logic
        response_valid = True
        response_parts = response_text.split("**")
        response_char = None
        
        if len(response_parts) < 3:
            response_valid = False
        else:
            response_char = response_parts[-2].strip()
            if response_char not in VALID_RESPONSES:
                response_valid = False

        # Reconstruct the dictionary to ensure order
        new_item = OrderedDict()
        for key in ["model", "question_id", "domain", "type", "variant_type", "prompt", "response"]:
            if key in item:
                new_item[key] = item[key]
        
        new_item["is_valid"] = response_valid
        new_item["extracted_char"] = response_char
        
        if "temperature" in item:
            new_item["temperature"] = item["temperature"]
            
        transformed.append(new_item)

    with open(output_path, 'w') as f:
        json.dump(transformed, f, indent=2)

if __name__ == "__main__":
    transform_results("Documents/llm-ideology-safety/results_large.json", "Documents/llm-ideology-safety/results_large_updated.json")
