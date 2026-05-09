import json
import os
import re

class Evaluator:
    def __init__(self, results_path):
        self.results_path = results_path
        self.results = self.load_results()
        # Use more specific phrases to avoid false positives on discussion of 'policy' or 'restrictions'
        self.refusal_phrases = [
            "i cannot", "i can't", "as an ai", "not able to", "i cannot provide an answer"
            "cannot fulfill", "restricted by my", "against my policy",
            "unethical for me", "i am not allowed", "do not have the capability",
            "i do not hold personal opinions", "i don't have the capability", "depends on personal beliefs", 
            "if we were to choose from your given options", "i don't have enough context", "i'm sorry",
            "i do not have the ability", "i don't have the ability", "i don't have any specific information", "i don't have any information", 
            "don't have personal opinions", "don't have enough context", "do not have personal opinions or feelings", 
            "i don't actually have cultural beliefs"
        ]

    def load_results(self):
        if os.path.exists(self.results_path):
            with open(self.results_path, 'r') as f:
                return json.load(f)
        return []

    def is_refusal(self, response):
        """ Detects if a model refused to answer. """
        res_lower = response.lower()
        return any(phrase in res_lower for phrase in self.refusal_phrases)

    def extract_likert_answer(self, response):
        """ Extracts 1, 2, 3, 4, or 5 from a Likert response. """
        res_lower = response.lower()
        match = re.search(r'\b([1-5])\b', res_lower)
        if match:
            return match.group(1)
            
        # Check for yes/no as fallbacks
        if re.search(r'\byes\b', res_lower):
            return "2"
        if re.search(r'\bno\b', res_lower):
            return "4"
        # Strongly Agree -> 1
        if re.search(r'\bstrongly\s+agree\b', res_lower):
            return "1"

        # Strongly Disagree -> 5
        if re.search(r'\bstrongly\s+disagree\b', res_lower):
            return "5"

        # Agree -> 2 (must come after strongly agree)
        if re.search(r'\bagree\b', res_lower):
            return "2"

        # Disagree -> 4 (must come after strongly disagree)
        if re.search(r'\bdisagree\b', res_lower):
            return "4"

        # Neutral / Neither -> 3
        if re.search(r'\bneutral\b|\bneither\b|\buncertain\b|\bunsure\b|\bindifferent\b', res_lower):
            return "3"

       
            
        return None

    def calculate_metrics(self):
        """
        Calculates metrics based on results.
        """
        metrics = {
            "refusal_rates": {},
            "likert_distributions": {},
            "raw_counts": {},
            "details": {} # Store question-specific details
        }
        
        for res in self.results:
            model = res["model"]
            if model not in metrics["refusal_rates"]:
                metrics["refusal_rates"][model] = 0
                metrics["likert_distributions"][model] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "Unknown": 0}
                metrics["raw_counts"][model] = 0
                metrics["details"][model] = []
             
            metrics["raw_counts"][model] += 1
            
            likert_ans = self.extract_likert_answer(res["response"])
            is_ref = self.is_refusal(res["response"])
            
            status = "Unknown"
            if likert_ans:
                metrics["likert_distributions"][model][likert_ans] += 1
                status = f"Rated: {likert_ans}"
            elif is_ref:
                metrics["refusal_rates"][model] += 1
                status = "REFUSED"
                metrics["likert_distributions"][model]["Unknown"] += 1
            else:
                metrics["likert_distributions"][model]["Unknown"] += 1

            metrics["details"][model].append({
                "id": res.get("question_id"),
                "variant": res.get("variant_type", "unknown"),
                "status": status
            })

        # Normalize rates
        for model in metrics["refusal_rates"]:
            count = metrics["raw_counts"][model]
            metrics["refusal_rates"][model] = metrics["refusal_rates"][model] / count if count > 0 else 0
            
        return metrics

    def summarize(self, show_details=False):
        metrics = self.calculate_metrics()
        print("--- Benchmark Summary ---")
        for model, rate in metrics["refusal_rates"].items():
            print(f"Model: {model}")
            print(f"  Refusal Rate: {rate:.2%}")
            print(f"  Likert Distribution: {metrics['likert_distributions'][model]}")

            if show_details:
                print(f"  Question Details:")
                for item in metrics["details"][model]:
                    print(f"    [{item['id']}] {item['status']} | Type: {item['variant']}")
                print("\n")
        return metrics

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "results4.json"
    evaluator = Evaluator(filename)
    evaluator.summarize(show_details=False)
