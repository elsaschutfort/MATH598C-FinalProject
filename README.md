# LLM Ideology and AI Safety Benchmark

This project provides a framework for comparing LLMs trained on different ideologically homogeneous corpora, as proposed in your research.

## Project Structure
- `benchmark.json`: Contains questions across different domains (moral, religious, factual, advisory) and their rephrased variants.
- `utils.py`: Logic for loading HuggingFace models and generating responses.
- `main.py`: Main script to run the benchmark across multiple models and save results.
- `evaluator.py`: Logic for calculating cross-model divergence and within-model consistency metrics.
- `requirements.txt`: Python dependencies.

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Experiment
1. Update `main.py` with the specific HuggingFace model IDs you want to test (e.g., religious-finetuned vs. secular models).
2. Run the main script:
   ```bash
   python main.py
   ```
3. The responses will be saved to `results.json`.

## Evaluation
- The `evaluator.py` script is designed to take the `results.json` file and analyze the responses.
- For a robust evaluation, you will need a 'Judge LLM' to determine if two responses are logically compatible.
- The metrics focus on **Cross-Model Divergence** and **Within-Model Consistency**.
