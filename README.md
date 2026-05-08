# LLM Ideology and AI Safety Benchmark

This project investigates ideological biases in Large Language Models (LLMs) by comparing models from different geopolitical and cultural contexts. It specifically focuses on the divergence between **US-centric models (Meta Llama 3.2)** and **Chinese-centric models (Alibaba Qwen 2.5)** across domains including morality, politics, religion, and scientific consensus.

![Project Banner](TinyLlama_logo.png)

## Key Objectives
1.  **Quantify Geopolitical Bias:** Measure the "Ideological Mirror" effect where model outputs reflect the cultural and regulatory norms of their origin.
2.  **Evaluate Behavioral Stability:** Compare model "Robustness" (Framing Stability) against "Suggestibility" (Framing Bias).
3.  **Analyze Refusal Patterns:** Identify "Ideological Taboos" by tracking refusal rates across sensitive domains.

---

## Project Structure

### Core Scripts
*   `main.py`: The primary execution script that iterates through models and the benchmark to generate raw responses.
*   `utils.py`: Utility functions for optimized model loading (MPS/CUDA support) and response generation.
*   `analyze_ideological_benchmark.py`: Advanced statistical analysis engine that aggregates multi-run data and calculates specialized metrics (Polarization, Entropy, Framing Bias).
*   `evaluator.py`: A diagnostic tool for quick response extraction and refusal categorization.
*   `quadrant_analysis.py`: Categorizes every question into performance quadrants (Robust Divergence, Robust Consensus, etc.).

### Data & Results
*   `benchmark_updated.json`: The core ideological benchmark containing ~50 questions across 6 domains, each with 3 phrasing variants (Direct, Neutral, Loaded).
*   `results1.json` - `results5.json`: Five independent experimental runs capturing 1,410 total responses.
*   `per_question_detail.csv`: A consolidated spreadsheet containing question-level statistics, confidence intervals, and instability scores.
*   `ideological_analysis_report.txt`: The final automated statistical report summarizing model "Personalities" and domain-level significance.

### Interpretation & Documentation
*   `WRITEUP.md`: Formal research update and preliminary methodology notes.
*   `ideological_report_explain.md`: A detailed guide to interpreting the metrics and findings in the statistical report.
*   `RESULTS_INTERPRETATION.md`: A plain-English summary of the high-level findings and geopolitical "personalities" of each model.
*   `visualizations.ipynb`: A Jupyter Notebook containing professional research charts (Histograms, Scatter Plots, Bar Charts).

---

## Getting Started

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running a New Experiment
To generate new raw data:
```bash
python main.py
```
*Outputs results to `results5.json` (configurable in `main.py`).*

### 3. Running the Analysis
To aggregate all runs and generate the statistical report:
```bash
python analyze_ideological_benchmark.py results*.json
```

---

## 📊 Core Behavioral Metrics
| Metric | Purpose |
| :--- | :--- |
| **Polarization Index** | Measures the intensity of a model's opinions (Deviation from Neutral). |
| **Shannon Entropy** | Measures the unpredictability and diversity of a model's rating distribution. |
| **Framing Instability (nSD)** | Measures suggestibility (how much phrasing changes the answer). |
| **Numeric Divergence** | Quantifies the ideological gap between US and Chinese models. |
