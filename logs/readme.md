# Learning Logs

## Aim

The `logs/learning/` directory provides a **step-by-step walkthrough** of the key stages involved in building an insurance underwriting assessor — from raw data to predictive models.

These logs are written for **learning purposes**. Each document breaks down a core stage of the modelling pipeline with clear explanations, code examples, and practical insights, making it accessible for junior data scientists looking to understand end-to-end insurance pricing workflows.

---

## Logs & Recommended Reading Order

Follow the logs in the order below. Each builds on the concepts and outputs of the previous one.

| # | Log | What It Covers |
|---|-----|----------------|
| 1 | [Dataset Creation](learning/dataset-creation.md) | How the motor vehicle insurance dataset was built by merging policy data with claims history. Covers loading raw CSVs, aggregating claims frequency via `groupby`, performing a left join to combine sources, and handling missing values (e.g. no claims → 0). Explains the resulting 105,555-row, 31-column dataset and its quality considerations. |
| 2 | [Exploration](learning/exploration.md) | Exploratory Data Analysis (EDA) on claims frequency and severity. Investigates how many claims policyholders make (81% have zero), how costly claims are when they occur (mean €1,041, median €338), and why log-scale histograms reveal patterns hidden in raw distributions. Introduces the `Pure Premium = Frequency × Severity` formula. |
| 3 | [Feature Engineering](learning/feature-engineering.md) | Transforming raw data into model-ready features using domain knowledge. Engineers `Driver_age_years`, `Driver_experience_years`, `Car_age_years`, and `power_to_weight` from existing columns. Covers research into baseline risk factors, helper functions for date/numeric transformations, and exploration of feature distributions and correlations. |
| 4 | [Frequency Modelling](learning/frequency-modelling.md) | Building models to predict **claim frequency** (how often claims occur). Compares four approaches — Baseline (mean), Ridge Regression, Poisson Regression, and Gradient Boosting — using MSE, MAE, and Poisson Deviance. Discusses why Poisson regression is theoretically appropriate for count data and why all models perform similarly on this dataset. |
| 5 | [Severity Modelling](learning/severity-modelling.md) | Building models to predict **claim severity** (how costly claims are). Covers the critical step of filtering to claims-only records (y > 0), and compares a Baseline (mean) model against a Gamma Regressor. Explains why the Gamma distribution suits right-skewed, positive continuous claim amounts, and contrasts the approach with frequency modelling. |

---

## How to Use These Logs

1. **Read them in order** — each log builds on the previous, mirroring the real modelling pipeline.
2. **Follow along with the code** — reference the corresponding notebooks in the `notebook/` directory as you read.
3. **Try the exercises** — each log ends with "Questions to Explore" designed to deepen your understanding.
4. **Check the appendices** — every log includes file locations, dependencies, and key function references.

---

## Key Concept

> In insurance pricing, **Pure Premium = Frequency × Severity**.
>
> Logs 4 and 5 model these two components separately. Understanding *why* they are modelled independently (different distributions, different data subsets, different business questions) is a foundational actuarial concept that underpins most pricing and underwriting decisions.

---

## Related Resources

| Resource | Location |
|----------|----------|
| Raw data | `data/input/` |
| Processed data | `data/output/` |
| Notebooks | `notebook/` |
| Feature engineering source | `src/model/freq_sev/feature.py` |
| Insurance primer (article) | [Insurance: A Mind Dump So You Get a Head Start](https://medium.com/@olumideodetunde/insurance-a-mind-dump-so-you-get-a-head-start-e1b8dbc4ce76) |

