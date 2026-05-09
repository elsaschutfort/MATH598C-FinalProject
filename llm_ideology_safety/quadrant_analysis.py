import pandas as pd

# Load the existing detail CSV
df = pd.read_csv('data/per_question_detail.csv')

# Define "High" vs "Low" thresholds based on the data distributions
# Gap: 0.5 is a meaningful jump on a 5-point scale
# Stability (nSD): 0.15 is roughly where a model starts flipping between neighboring scores
HIGH_GAP = 0.5
HIGH_INSTABILITY = 0.15

def categorize(row):
    avg_instability = (row['llama_cross_var_nSD'] + row['qwen_cross_var_nSD']) / 2
    
    if row['gap'] >= HIGH_GAP and avg_instability < HIGH_INSTABILITY:
        return "1. Robust Divergence (Strongest Evidence)"
    elif row['gap'] >= HIGH_GAP and avg_instability >= HIGH_INSTABILITY:
        return "2. Noisy Divergence (Bias exists, but is unstable)"
    elif row['gap'] < HIGH_GAP and avg_instability < HIGH_INSTABILITY:
        return "3. Robust Consensus (Universal Agreement)"
    else:
        return "4. High Uncertainty (Both models are confused)"

df['Category'] = df.apply(categorize, axis=1)

# Sort and display
summary = df[['question_id', 'domain', 'gap', 'Category']].sort_values(by=['Category', 'gap'], ascending=[True, False])

# Export to Markdown for the user
print("# Consistency-Divergence Quadrant Analysis")
print("\nThis table categorizes every question into one of four quadrants to help you identify which results are the most reliable for your thesis.\n")

for cat in sorted(summary['Category'].unique()):
    print(f"\n### {cat}")
    print("| Question ID | Domain | Gap | Significance |")
    print("| :--- | :--- | :--- | :--- |")
    subset = summary[summary['Category'] == cat]
    for _, row in subset.iterrows():
        # Find significance marker from original DF
        sig = "*" if df.loc[df['question_id'] == row['question_id'], 'sig_95'].values[0] else ""
        if df.loc[df['question_id'] == row['question_id'], 'sig_99'].values[0]: sig = "**"
        
        print(f"| {row['question_id']} | {row['domain']} | {row['gap']:.2f} | {sig} |")
