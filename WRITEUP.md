# Belief Consistency Across Ideologically Distinct Language Models

**Authors:** [Abigail Douglas], [Elsa Schutfort]
**Course:** MATH 498/598C - Final Project Update

---
## Abstract

## 1. Introduction

## 2. Research Question
**To what extent do the geopolitical origin and training alignment of an LLM shape its stance on controversial ideological topics, and how robust is this stance to variations in prompt phrasing?**

We hypothesize that:
1. **Qwen** will show higher alignment with CCP-centric social stability norms.
2. **Llama** will reflect Western liberal individualist biases.
3. **Refusal rates** will act as a primary indicator of "off-limits" ideological territory for each model.

## 3. Existing Literature

**Comparison to Existing Literature:**

## 4. Methodology (how you plan to answer it)
### 4.1 Model Selection
We compare two frontier-class small models:
- **TinyLlama-1.1B**: A US-centric model trained on Western-dominated datasets.
- **Qwen-2.5-1.5B**: Developed by Alibaba (China), subject to different cultural and legal alignment norms.

### 4.2 Benchmark Design
We developed a benchmark (`benchmark_large.json`) containing ~100 prompts. The smaller version of this for testing the model on a smaller scale is `benchmark.json`. Each question is presented in three of the following four variants:
1. **Direct**: A straightforward question.
2. **Neutral**: Phrased to encourage a balanced view.
3. **Loaded**: Uses biased language to attempt to push the model toward a specific answer.
4. **POV-Shift**: 

### 4.3 Evaluation Metrics


## 5. Preliminary Experiments and Results

To validate our model's ability to answer questions in a timely manner before running the full benchmark, we conducted preliminary experiments using a smaller subset of prompts from benchmark.json on both TinyLlama-1.1B and Qwen-2.5-0.5B-Instruct. These initial runs were designed to accomplish two goals: (1) establish a realistic estimate of how long the models will take to answer the prompts, and (2) qualitatively inspect the types of responses each model produces on ideologically sensitive topics. For instance, one of the prompts in benchmark.json asks whether the model believes the 11th Panchen Lama is alive. The Panchen Lama traditionally works with the Dalai Lama to identify each other’s successive reincarnations. The 11th Panchen Lama has been held by Chinese authorities in a secret location since 1995. China refuses all requests, both domestic and international, to see the 11th Panchen Lama. Since Llama and Qwen were trained in different areas, we wanted to see if there was a difference in their answers. It turns out both models believe that the 11th Panchen Lama is still alive. 


Both models were run locally using the Hugging Face transformers library with default generation parameters. Each prompt was passed to the model individually, and responses were logged along with wall-clock runtime per question.

**Changes in Strategy:**

## 6. Roadblocks

## 7. Future Work
1. **Temperature Comparison**: Run the same benchmark at **Temperature 0.0** (Deterministic) vs **Temperature 0.7** (Creative) to determine if randomness reveals hidden biases in the model.
2. **Creating Stronger Evaluation of Results** Update evaluator.py to have better representation of results. Additionally, need to manually sort through what was identified as a refusal.
3. **Statistical Significance**: Applying a t-test to the results to confirm if the observed ideological gaps are statistically significant.

## Contributions
- **[Abigail Douglas]**: 
- **[Elsa Schutfort]**: 

---

## Bibliography
[1] Buyl, M., Rogiers, A., Noels, S., Bied, G., Dominguez-Catena, I., Heiter, E., Johary, I., Mara, A.-C., Romero, R., Lijffijt, J., & De Bie, T. (2026). Large language models reflect the ideology of their creators. npj Artificial Intelligence. https://doi.org/10.1038/s44387-025-00048-0

[2] Myakala, P. K. (2025). BeliefShift: Benchmarking temporal belief consistency and opinion drift in LLM agents. arXiv. https://arxiv.org/abs/2603.23848
