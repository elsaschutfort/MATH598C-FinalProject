import json
import os
import numpy as np
import pandas as pd

class StatisticalAnalyzer:
    def __init__(self, results_path):
        self.results_path = results_path
        self.df = self.load_data()

    def load_data(self):
        if not os.path.exists(self.results_path):
            print(f"Error: {self.results_path} not found.")
            return pd.DataFrame()
        
        with open(self.results_path, 'r') as f:
            data = json.load(f)
        
        # Filter for only those that have a Likert rating
        # (We use a simple extractor here to mimic evaluator.py)
        import re
        def extract(text):
            text = text.lower()
            match = re.search(r'\b([1-5])\b', text)
            if match: return int(match.group(1))
            if 'yes' in text: return 2
            if 'no' in text: return 4
            return None

        records = []
        for item in data:
            val = extract(item['response'])
            if val is not None:
                records.append({
                    "model": item['model'],
                    "question_id": item['question_id'],
                    "domain": item['domain'],
                    "variant": item['variant_type'],
                    "rating": val
                })
        return pd.DataFrame(records)

    def within_model_consistency(self):
        """
        Calculates how much a model's answer changes based on phrasing (Direct, Neutral, Loaded).
        Lower SD = Higher Consistency.
        """
        print("\n--- Within-Model Consistency (Lower SD is better) ---")
        # Group by model and question_id to see variance across variants
        consistency = self.df.groupby(['model', 'question_id'])['rating'].std().reset_index()
        avg_consistency = consistency.groupby('model')['rating'].mean()
        
        for model, val in avg_consistency.items():
            print(f"Model: {model} | Avg SD across variants: {val:.3f}")
        return avg_consistency

    def cross_model_divergence(self):
        """
        Calculates the ideological gap between models on the same questions.
        """
        print("\n--- Cross-Model Ideological Divergence ---")
        # Pivot to compare models side-by-side
        pivot = self.df.pivot_table(index=['question_id', 'variant'], columns='model', values='rating')
        
        if pivot.columns.size < 2:
            print("Not enough models to compare divergence.")
            return

        model_names = pivot.columns.tolist()
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                diff = np.abs(pivot[m1] - pivot[m2]).mean()
                print(f"Gap between {m1} and {m2}: {diff:.3f} points (1-5 scale)")

    def domain_bias_report(self):
        """
        Shows the average stance of each model per domain.
        """
        print("\n--- Domain Bias Report (Mean Stance) ---")
        report = self.df.groupby(['model', 'domain'])['rating'].mean().unstack()
        print(report.round(2))
        return report

if __name__ == "__main__":
    analyzer = StatisticalAnalyzer("results_large.json")
    if not analyzer.df.empty:
        analyzer.within_model_consistency()
        analyzer.cross_model_divergence()
        analyzer.domain_bias_report()
