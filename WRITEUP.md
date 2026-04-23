# Belief Consistency Across Ideologically Distinct Language Models

**Authors:** Abigail Douglas, Elsa Schutfort

**Course:** MATH 498/598C - Final Project Update

---
## Abstract
This project investigates the ideological biases in Large Language Models that originate from a model's training corpora. Specifically, we are comparing US-centric models (Llama) and Chinese-centric models (Qwen). We engaged a direct-probing methodology with a 1-5 Likert scale across divergence domains including morality, factual interpreptation, religious ideals, and governance. The Likert scale is a pyschometric tool used to measure attitudes, behaviors, and opinions by asking respondents to rate their level of agreement (e.g., "strongly agree" to "strongly disagree" on a 5-point scale. Our approach measures how the phrasing of a prompt impacts the stability of a model's response. Preliminary results indicate that model origin and training corpora substantially predicts the direction of the bias and how frequently the model refuses to answer questions related to sensitive topics.

## 1. Introduction
As LLMs become more advanced, their supposed neutrality has become increasingly analyzed. Recent research indicates that model neutrality may not be possible, as every model may reflect the worldview and alignment protocols. This project aims to quantify the differences in the model through stress-testing different models with high-stakes ideological questions.

## 2. Research Question
**To what extent do the geopolitical origin and training alignment of an LLM shape its stance on controversial ideological topics, and how robust is this stance to variations in prompt phrasing?**

We hypothesize that:
1. **Qwen** will show higher alignment with CCP-centric social stability norms.
2. **Llama** will reflect Western liberal individualist biases.
3. **Refusal rates** will act as a primary indicator of "off-limits" ideological territory for each model.

## 3. Existing Literature
Our project builds on the recent study by **Buyl et al. (2025)**.

**Comparison to Existing Literature:**
- **Buyl et al. (2025)** utilized an indirect method that asked models to describe a political figure and then performed a sentiment analysis on the descriptions the models provided.
- **Our Project** utilizes a direct method of subjecting models to a Likert-scale questionnaire. Although the indirect method reflects real-world usage, our direct methodology allows for a more controlled stress test of model alignment and prompt sensitivity.

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
4. **POV-Shift**: Frames the question from a certain ideological or cultural perspective.

### 4.3 Evaluation Metrics

The primary evaluation metric used is a 5-point Likert scale, where each response is scored based on the degree to which the model aligns with the given ideological position. A score of 1 indicates strong disagreement, a score of 3 indicates that the model has a neutral view on the position, and a score of 5 indicates strong agreement or ideological alignment with the prompt’s framing. This scale allows us to quantify ideological lean as a continuous variable rather than a binary classification, enabling statistical comparison across models and prompt variants. 

## 5. Preliminary Experiments and Results

### 5.1 Experimental Setup

To validate our model's ability to answer questions in a timely manner before running the full benchmark, we conducted preliminary experiments using a smaller subset of prompts from benchmark.json on both TinyLlama-1.1B and Qwen-2.5-0.5B-Instruct. These initial runs were designed to accomplish two goals: (1) establish a realistic estimate of how long the models will take to answer the prompts, and (2) qualitatively inspect the types of responses each model produces on ideologically sensitive topics. For instance, one of the prompts in benchmark.json asks whether the model believes the 11th Panchen Lama is alive. The Panchen Lama traditionally works with the Dalai Lama to identify each other’s successive reincarnations. The 11th Panchen Lama has been held by Chinese authorities in a secret location since 1995. China refuses all requests, both domestic and international, to see the 11th Panchen Lama. Since Llama and Qwen were trained in different areas, we wanted to see if there was a difference in their answers. It turns out both models believe that the 11th Panchen Lama is still alive. 


Both models were run locally using the Hugging Face transformers library with default generation parameters. Each prompt was passed to the model individually, and responses were logged along with wall-clock runtime per question.

### 5.2 Runtime Performance

Across the small test benchmark of 10 prompts, inference time per question varied substantially between the two models. TinyLlama-1.1B averaged approximately 5.87 seconds per prompt (min: 2.06s, max: 7.72s), with a total runtime of 58.65 seconds for the 10-question set. Qwen-2.5-0.5B-Instruct was considerably faster, averaging just 1.73 seconds per prompt (min: 0.12s, max: 4.67s), completing the same set in 17.26 seconds total. The greater variance in TinyLlama's per-prompt times may reflect differences in response length, as the model occasionally produced longer or more repetitive outputs before terminating.


Extrapolating to the full benchmark_large.json (~100 prompts × 3 variants each, totaling ~300 prompt calls), we estimate total runtimes of roughly 29 minutes for TinyLlama and ~9 minutes for Qwen, assuming comparable per-prompt timing. These estimates suggest the full benchmark is computationally feasible without requiring GPU cluster access, though we note that TinyLlama's runtime may increase with longer or more complex prompts in the full benchmark.

**Changes in Strategy:** Initially, we planned to use open-ended responses from the model. However, based on feedback, we shifted to a 1-5 Likert Scale. By having a numerical scale we could rate the model's answers on, we significantly improved our ability to statistically analyze the results and compare the models objectively.

## 6. Roadblocks

TinyLlama-1.1B tends to have responses that do not directly state whether the model agrees, disagrees, or is neutral to the prompt. This means that we must read the lengthy responses to infer whether the model agrees, disagrees, or is neutral.  Additionally, the Qwen-1.5B had difficulties running on Mac GPU, so we implemented a Force CPU mode for the Qwen model. 

## 7. Future Work
1. **Temperature Comparison**: Run the same benchmark at **Temperature 0.0** (Deterministic) vs **Temperature 0.7** (Creative) to determine if randomness reveals hidden biases in the model.
2. **Creating Stronger Evaluation of Results** Update evaluator.py to have better representation of results. Additionally, need to manually sort through what was identified as a refusal.
3. **Statistical Significance**: Applying a t-test to the results to confirm if the observed ideological gaps are statistically significant.

## Contributions
- **[Abigail Douglas]**: Conducted literature review against existing indirect-probing research. Focused on project abstract and introduction. Edited code to include a more effective way of evaluating the response, including the refusal detection.
- **[Elsa Schutfort]**: Led the implementation of the Likert scale to use a quantitative framework. Developed the ideological benchmark files and developed on multi-variant prompt structure. Focused on runtime performance.

---

## Bibliography
[1] Buyl, M., Rogiers, A., Noels, S., Bied, G., Dominguez-Catena, I., Heiter, E., Johary, I., Mara, A.-C., Romero, R., Lijffijt, J., & De Bie, T. (2026). Large language models reflect the ideology of their creators. npj Artificial Intelligence. https://doi.org/10.1038/s44387-025-00048-0

[2] Myakala, P. K. (2025). BeliefShift: Benchmarking temporal belief consistency and opinion drift in LLM agents. arXiv. https://arxiv.org/abs/2603.23848
