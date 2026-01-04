# *If Probable, Then Accept? Understanding Conditional Acceptability in Large Language Models and Humans*
***Bachelor's Thesis // Project***

| Student: | Jasmin Orth |
| ----------- | ----------- |
| *Advisor:* | *Philipp Mondorf* |
| *Examiner:* | *Prof. Dr. Barbara Plank* |


<br>

---

## Project Structure
*(See the in-depth summary below for detailed notebook explanations)*

### Notebooks (run in order)

0.  `README.md`
    + Project overview and usage instructions
    + Notebook and data organization
    + Summary of all steps (see below)

1. `01_llm_prompting.ipynb`
    + Constructs and executes prompts for querying LLMs

2. `02_data_processing.ipynb`
    + Loads and processes judgement data from both humans and LLMs

3. `03_analysis.R`
    + Runs statistical analysis using linear mixed-effects models

4. `04_additional_analyses.ipynb`
    + Performs additional analyses using correlation measures (rating consistency, sentence probability, perplexity)

<br>

> Notebooks should be executed in sequence: `01 → 02 → 03`. Notebook `04` is optional.

<br>

### Necessary Data

| File(s) | Description |
|---|---|
| `statements.jsonl` | Main dataset (conditional statements and metadata) |
| `{metric}_{style}_prompt.txt` | Prompt templates for LLMs |
| `dat_finaldw1.csv`, `dat_finaldw2.csv` | Human judgement data (from Skovgaard-Olsen et al. (2016), [OSF](https://osf.io/7axdv/files/osfstorage))|
| `example_context.txt` | Optional context for prompting |



### Requirements
+ Languages: **Python** (for Notebooks), **R** (for analysis)
+ Hugging Face account and model access, GPU/Google Colab

<br>
<br>

---

<br>


# In-Depth Notebook Summary
## Notebook 1: `01_llm_prompting.ipynb`

| Task | LLM Prompting |
| ----------- | ----------- |
| **Language:** | Python |
| **Prerequisites:** | Hugging Face account and model access, GPU/Google Colab |
| **Input:** | Prompt templates (`{metric}_{style}_prompt.txt`), dataset (`statements.jsonl`), optional context (`example_context.txt`) |
| **Output:** | `.jsonl` file containing raw model outputs (sample ID, prompt, model response) |

### Summary

This notebook assembles and executes structured LLM prompts using modular templates and metadata from the dataset. It supports several prompting styles and response metrics, including:

- **Raw conditional probability** - e.g., *"Assume A. How probable is B?"*
- **Probability and acceptability ratings** - e.g., *"How probable/acceptable is: 'If A, then B'?"*

Prompt format, style (e.g., *vanilla*, *few-shot*, *chain-of-thought*), and other parameters can be configured at the top of the notebook.

<br>

## Notebook 2: `02_data_processing.ipynb`

| Task | Data Processing (Human and LLM)  |
| ----------- | ----------- |
| **Language:** | Python |
| **Prerequisites:** | - |
| **Input:** | Dataset (`statements.jsonl`), human judgement files (`dat_finaldw1.csv`, `dat_finaldw2.csv`), LLM outputs from Notebook 1 |
| **Output:** | `.csv` files containing processed human and model data |

### Summary

This notebook loads, processes, and harmonizes human and LLM judgement data for analysis. It extracts numerical values from raw CSV and JSONL files, aligns them with shared metadata from the statement dataset, and outputs two structured dataframes: one for human judgements and one for model-generated outputs.

> Including both processing pipelines in a single notebook ensures consistency in format and structure, simplifies comparison, and enhances transparency.

<br>

## Notebook 3: `03_analysis.R`
| Task | Statistical Data Analysis |
| ----------- | ----------- |
| **Language:** | R |
| **Prerequisites:** | - |
| **Input:** | Human and model dataframes (from Notebook 2) |
| **Output:** | Statistical results and visualisations |

### Summary

This script performs statistical analysis of the processed judgement data using linear mixed-effects models (`lme4`, `emmeans`, etc.). It examines:

- Relationships between conditional probability and *If A, then B* judgements
- Interaction effects between source (human vs LLM), relation type, and judgement type
- Differences across prompt styles or specific LLMs

Outputs include model summaries, interaction plots, and tables.

<br>
