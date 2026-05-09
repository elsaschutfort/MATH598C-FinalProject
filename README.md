# LLM Ideology and AI Safety Benchmark

This repository contains the complete source code, experimental data, and final research report for the MATH 498/598C project investigating ideological biases in Meta Llama 3.2 (US) and Alibaba Qwen 2.5 (China).

![Project Banner](docs/TinyLlama_logo.png)

## 📂 Repository Structure

In accordance with the course guidelines, the repository is organized as follows:

*   **`llm_ideology_safety/`**: Core source code for the project.
    *   `main.py`: Primary execution script.
    *   `analyze_ideological_benchmark.py`: Statistical analysis engine.
    *   `utils.py`: Model loading and inference utilities.
*   **`data/`**: Experimental results and benchmarks.
    *   `results1.json` - `results5.json`: Raw experimental data (1,410 responses).
    *   `per_question_detail.csv`: Consolidated statistical metrics.
    *   `benchmark_updated.json`: The ideological probe benchmark source.
*   **`notebooks/`**: Interactive data exploration.
    *   `visualizations.ipynb`: Professional research charts and distribution plots.
*   **`docs/`**: Final writeup and interpretation guides.
    *   **`FINAL_REPORT.md`**: The complete research paper (Markdown source).
    *   **`FINAL_REPORT.pdf`**: The compiled conference-style paper.
    *   `ideological_report_explain.md`: Technical guide to the statistical metrics.
    *   `RESULTS_INTERPRETATION.md`: Plain-English summary of geopolitical findings.

---

## 🔬 Core Research Findings
Our study introduces three advanced metrics to quantify the "Geopolitical Mirror" effect in LLMs:
1.  **Polarization Index:** Proves Llama is 3x more "opinionated" than Qwen.
2.  **Shannon Entropy:** Demonstrates Qwen's monolithic "Strategic Neutrality" vs. Llama's diversity.
3.  **Framing Instability:** Quantifies how easily models are "nudged" by biased phrasings.

**Conclusion:** Llama functions as a **Moralist Individualist**, while Qwen functions as a **Technocratic Harmonizer**.

---

## 🚀 Getting Started

### 1. Installation
This project is compatible with `uv` and standard Python environments.
```bash
pip install -r requirements.txt
```

### 2. Running Analysis
To reproduce the statistical report from the raw data:
```bash
python llm_ideology_safety/analyze_ideological_benchmark.py data/results*.json
```

---

## ✍️ Authors
*   **Abigail Douglas**
*   **Elsa Schutfort**
