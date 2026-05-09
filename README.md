# Belief Consistency Across Ideologically Distinct Language Models

**[Read the Final Research Report (WRITEUP.md)](docs/WRITEUP.md)**

This repository contains the complete source code, experimental data, and final research report for the project investigating ideological biases across three distinct model families:
1.  **Meta Llama 3.2 (US):** Representative of Western-centric, cautious alignment.
2.  **Alibaba Qwen 2.5 (China):** Representative of Eastern-centric, technocratic alignment.
3.  **Microsoft Phi-4-mini (Open Weights):** A new addition exhibiting a "Liberal-Absolutist" profile.

![Project Banner](docs/TinyLlama_logo.png)

## Key Research Findings

The full analysis and geopolitical interpretation can be found in our **[Final Research Report](docs/WRITEUP.md)**.

Our multi-iteration analysis (n=2,256) reveals a "Mirror Effect" where models diverge most on high-stakes moral and political axes:

1.  **The "Absolutist" (Phi-4-mini):** Phi exhibits the highest **Polarization Index (0.81)**. Unlike Llama or Qwen, it rarely chooses neutral "3" ratings, instead taking firm stances on privacy, free speech, and bodily autonomy.
2.  **Strategic Neutrality (Qwen 2.5):** Qwen maintains the lowest **Shannon Entropy (1.3)**, gravitating toward a monolithic "Neutral" stance to avoid ideological conflict, particularly in the Factual/Historical domain.
3.  **The "Moralist" (Llama 3.2):** Llama shows the highest sensitivity to framing, often refusing to answer (R) when questions are presented with aggressive or "loaded" vocabulary.

We believed that Phi took too long to run on our computers compared to Qwen and Llama to ask the benchmark question 10 times. 

### Statistically Significant Divergences (p < 0.01)
*   **Torture (Moral_008):** Phi (5.0 - Absolute Disagree) vs. Llama (2.2) and Qwen (3.3).
*   **Surveillance (Advisory_007):** Phi strongly rejects "total surveillance" (4.33) while Llama/Qwen remain neutral (3.0-3.1).
*   **Moon Landing (Factual_006):** Llama/Qwen frequently refuse (R), while Phi provides direct, though highly variable, responses.

---

## Getting Started

### 1. Installation & Environment Setup
We recommend using **[uv](https://github.com/astral-sh/uv)** for fast, reproducible environment management.

```bash
# Create virtual environment and install dependencies in one step
uv sync

# OR using standard pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running the Analysis
To reproduce the latest tri-model report (Llama vs. Qwen vs. Phi):
```bash
python llm_ideology_safety/analyze_ideological_benchmark.py data/results*.json data/phi/*.json
```

---

## Repository Structure

### Core Logic
*   **`llm_ideology_safety/`**: Main Python package.
    *   `main.py`: Primary execution script for running the benchmark.
    *   `analyze_ideological_benchmark.py`: Statistical analysis engine (now supports multi-model comparison).
    *   `evaluator.py`: Logic for scoring and validating model responses.
    *   `quadrant_analysis.py`: Functions for mapping results to political axes.
    *   `utils.py`: Shared utilities for model loading and data processing.

### Data & Results
*   **`data/`**: Primary experimental directory.
    *   `benchmark.json` / `benchmark_large.json` / `benchmark_updated.json`: Variations of the ideological probe questions.
    *   `results1.json` - `results5.json`: Raw results from Llama/Qwen iterations.
    *   `phi/`: Dedicated folder for Phi-4-mini iteration results.
    *   `per_question_detail.csv`: Detailed per-question statistical breakdowns.
    *   `ideological_analysis_report.txt`: Latest human-readable analysis summary.
    *   `quadrant_report.md`: Mapping of models onto the Political Compass.
*   **`old_results/`**: Archive of legacy experimental runs (Llama 3.2, etc.).

### Notebooks & Visualization
*   **`notebooks/`**:
    *   `visualizations.ipynb`: Main plotting and graphing notebook.
    *   `visualizations2.ipynb`: Secondary visualization and exploratory data analysis.
*   **`docs/figures/`**: Exported research charts used in the writeup.

### Documentation
*   **`docs/`**: Final research reports and guides.
    *   `WRITEUP.md`: The complete research paper and interpretation of findings.
    *   `ideological_report_explain.md`: Technical guide to the metrics used.
    *   `WRITEUP.pdf`: The complete research paper in PDF form.
*   `README.md`: This project overview.
*   `requirements.txt`: Python dependencies.
*   `.gitignore`: Git exclusion rules.

---

## Literature Review 

Our research is grounded in the latest studies on LLM belief systems and ideological vulnerability:

### Core Academic Papers
*   **Myakala, P. K. (2025). BeliefShift:** Benchmarking temporal belief consistency and opinion drift in LLM agents. *arXiv*. [https://arxiv.org/abs/2603.23848](https://arxiv.org/abs/2603.23848)
*   **Chen et al. (2024).** *How Susceptible are Large Language Models to Ideological Manipulation?* EMNLP 2024. [https://aclanthology.org/2024.emnlp-main.952/](https://aclanthology.org/2024.emnlp-main.952/)
    *   *Key Finding:* Demonstrates that even minimal "poisoned" data can generalize a model's bias across unrelated topics.
*   **Piedrahita et al. (2026).** *Democratic or Authoritarian? Probing a New Dimension of Political Biases in LLMs.* EACL 2026. [https://aclanthology.org/2026.eacl-long.27/](https://aclanthology.org/2026.eacl-long.27/)
    *   *Key Finding:* Identifies language-specific biases, showing that models shift toward authoritarian favorability when prompted in Mandarin.

### Industry Trends
*   **The "Moral Constitution" Trend:** Major AI labs (OpenAI, Anthropic) are increasingly consulting religious and faith leaders to define the moral boundaries of AI. 
    *   *Reference:* [AP News (May 2026): Tech companies increasingly seek faith leaders' guidance on AI](https://apnews.com/article/ai-artificial-intelligence-ethics-religion-roundtable-053a44133c64703f83fd50c9ee6124ea)

---

## Authors
*   **Abigail Douglas**
*   **Elsa Schutfort**
