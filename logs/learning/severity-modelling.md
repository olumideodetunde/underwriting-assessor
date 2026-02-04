# Severity Modelling: Predicting Insurance Claim Amounts

## 1. Introduction

Welcome! This guide walks you through how we build **severity models** to predict how much a claim will cost. This is the second critical component of insurance pricing, complementing frequency modelling.

### What is Severity Modelling?

Severity modelling predicts the **monetary amount** of claims when they occur. In insurance mathematics:

```
Premium = Frequency × Severity
```

Where:
- **Frequency**: How many claims (covered in frequency modelling)
- **Severity**: How much each claim costs (what we're modelling here)

### Purpose

By accurately predicting claim severity, insurers can:
- Estimate the financial impact of claims
- Set appropriate reserves for outstanding claims
- Price policies to cover expected losses
- Identify factors that drive claim costs

### Real-World Context

Consider two motor insurance claims:
- A minor fender-bender in a parking lot
- A multi-vehicle collision on the motorway

Both count as one claim (frequency = 1), but their costs differ dramatically. Severity models capture these differences based on:
- Vehicle value and type
- Geographic area (repair costs vary by location)
- Type of coverage and risk category

### Key Difference from Frequency Modelling

| Aspect | Frequency | Severity |
|--------|-----------|----------|
| Target | Claim count (0, 1, 2, ...) | Claim amount (£500, £2,000, ...) |
| Data used | All policies | Only policies with claims |
| Distribution | Poisson (count data) | Gamma (positive continuous) |
| Zero values | Many zeros expected | Zeros excluded |

---

## 2. Data Preparation

### Step 1: Load Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import GammaRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

### Step 2: Create the Dataset

We use the same data preparation as frequency modelling:

```python
# Load raw data
insurance = pd.read_csv('../../data/input/Motor_vehicle_insurance_data.csv', delimiter=";")
claims = pd.read_csv('../../data/input/sample_type_claim.csv', delimiter=';')

# Aggregate claims to get frequency per policyholder per year
claims_frequency = (
    claims
    .groupby(['ID', 'Cost_claims_year'])
    .agg({'Cost_claims_by_type': 'count'})
    .rename(columns={'Cost_claims_by_type': 'claims_frequency'})
    .reset_index()
)

# Merge with insurance data
dataset = (
    pd
    .merge(
        left=insurance,
        right=claims_frequency,
        how='left',
        on=['ID', 'Cost_claims_year']
    )
    .fillna(value={'claims_frequency': 0})
)
```

### Step 3: Split into Training and Test Sets

```python
trainset, testset = train_test_split(
    dataset, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    shuffle=True        # Randomize the split
)
```

| Step | Action |
|------|--------|
| Load | Read insurance and claims data |
| Aggregate | Count claims by (ID, year) |
| Merge | Left join on (ID, year) |
| Fill | NaN → 0 for no claims |
| Split | 80/20 train-test split |

---

## 3. Feature Engineering

Features are created using the same dedicated feature engineering module:

```python
from src.model.freq_sev.feature import main as feature_main

features_trainset = feature_main(trainset)
features_testset = feature_main(testset)
```

The feature engineering process transforms raw data into predictive features suitable for modelling. See the feature engineering documentation for details.

---

## 4. Understanding the Target Variable

Before modelling, we examine the distribution of our target variable `Cost_claims_year` on the **training set only**.

### Why Cost_claims_year?

The response variable represents the **total claim amount per policy per year**. Unlike frequency modelling where we predict counts, severity modelling predicts monetary amounts.

### Key Considerations for Severity Modelling

1. **Distribution shape**: Claim amounts are typically right-skewed (many small claims, few large ones)
2. **Positive values only**: We only use policies where claims occurred (severity > 0)
3. **Appropriate distribution**: Gamma distribution suits positive continuous outcomes

### Distribution Analysis

```python
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 5))

# Full distribution (log scale for visibility)
ax0.set_title('Loss Distribution')
features_trainset['Cost_claims_year'].hist(bins=40, log=True, ax=ax0)

# Middle 95% to see the bulk of claims
p2_5, p97_5 = np.percentile(features_trainset['Cost_claims_year'], [2.5, 97.5])
middle_95 = features_trainset['Cost_claims_year'][
    (features_trainset['Cost_claims_year'] >= p2_5) & 
    (features_trainset['Cost_claims_year'] <= p97_5)
]
ax1.set_title('Middle-95% Loss Distribution (2.5%-97.5%)')
middle_95.hist(bins=40, log=False, ax=ax1)

print(f"Average loss distribution: {np.average(features_trainset['Cost_claims_year'])}")
```

**Key Findings**:
- The distribution is heavily right-skewed with a long tail of high-value claims
- The middle 95% shows the bulk of claims are concentrated at lower values
- This confirms the Gamma distribution is an appropriate choice for modelling

### Why the Right-Skewed Distribution Matters

Most claims are relatively small (minor repairs, small incidents), but a few claims can be extremely expensive (total losses, serious accidents). This "long tail" is characteristic of insurance claim data and is why we use the Gamma distribution rather than a normal distribution.

---

## 5. Model Development

We develop and compare two models.

### Selected Features

```python
train_variables = [
    'Car_age_years',         # Vehicle age
    'Type_risk',             # Vehicle usage type
    'Area',                  # Geographic location
    'Value_vehicle',         # Vehicle monetary value
    'Distribution_channel',  # How policy was purchased
    'Cylinder_capacity'      # Engine size
]
target = ['Cost_claims_year']
```

### Evaluation Metrics

```python
def model_evaluation_metrics(estimator, df_test, target_variable, training_variables):
    y_pred = estimator.predict(df_test[training_variables])
    mse = mean_squared_error(df_test[target_variable], y_pred)
    mae = mean_absolute_error(df_test[target_variable], y_pred)
    print(f"MSE: {mse:.3f}")
    print(f"MAE: {mae:.3f}")
```

**Why these metrics?**
- **MSE (Mean Squared Error)**: Penalizes large errors heavily — important for severity where big misses are costly
- **MAE (Mean Absolute Error)**: Average absolute prediction error in the same units as claim amounts

### Critical Step: Filter for Policies with Claims

```python
# Only keep records where claims occurred
train_mask = features_trainset['Cost_claims_year'] > 0
updated_features_trainset = features_trainset[train_mask]

test_mask = features_testset['Cost_claims_year'] > 0
updated_features_testset = features_testset[test_mask]
```

**Why filter for claims > 0?**

This is a crucial difference from frequency modelling:
- The Gamma distribution only supports strictly positive values (y > 0)
- We're modelling "given a claim happens, how much will it cost?"
- Policies with no claims have no severity to predict

| Dataset | Purpose |
|---------|---------|
| Full dataset | Used for frequency modelling (includes zeros) |
| Filtered dataset (claims > 0) | Used for severity modelling |

---

## 6. Model 1: Baseline (Mean Prediction)

The simplest model: predict the average claim amount for everyone.

```python
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
dummy_regressor = dummy.fit(
    updated_features_trainset[train_variables], 
    updated_features_trainset[target]
)
```

**Results**:
| Metric | Value |
|--------|-------|
| MSE | (baseline) |
| MAE | (baseline) |

**Interpretation**: This is our benchmark. Any useful model should beat these numbers. The baseline predicts the same average severity for all claims regardless of policy characteristics.

---

## 7. Model 2: Gamma Regressor

The Gamma Regressor is specifically designed for positive continuous data like claim amounts.

```python
from sklearn.linear_model import GammaRegressor

gamma = GammaRegressor(
    alpha=10,                  # Regularization strength
    solver="newton-cholesky",  # Optimization algorithm
    max_iter=10000             # Maximum iterations
)
gamma_regressor = gamma.fit(
    updated_features_trainset[train_variables], 
    updated_features_trainset[target].values.ravel()
)
```

**Why Gamma Regression?**
- Assumes target follows a Gamma distribution (appropriate for positive continuous values)
- Naturally handles right-skewed data
- Log link function ensures predictions are always positive
- Well-suited for modelling claim amounts, repair costs, and similar financial data

**Results**:
| Metric | Value |
|--------|-------|
| MSE | (evaluate) |
| MAE | (evaluate) |

**Interpretation**: Compare these metrics to the baseline to assess whether the model captures meaningful patterns in the data.

---

## 8. Model Evaluation

### Model Comparison

```python
for model in [dummy_regressor, gamma_regressor]:
    print(f"Now evaluating model {model.__class__.__name__}")
    model_evaluation_metrics(
        estimator=model, 
        df_test=updated_features_testset, 
        target_variable=target, 
        training_variables=train_variables
    )
    print("-------------")
```

### Summary Table

| Model | MSE | MAE | Description |
|-------|-----|-----|-------------|
| Baseline (Mean) | - | - | Predict average severity |
| Gamma Regressor | - | - | Severity-specific GLM |

### Visual Comparison: Observed vs Predicted

Visualizing how well predictions align with actual values:

```python
def plot_obs_pred(df, feature, observed, predicted, y_label=None, title=None, ax=None):
    df_ = df.loc[:, [feature]].copy()
    df_["observed"] = df[observed]
    df_["predicted"] = predicted
    df_ = (
        df_.groupby([feature])[["observed", "predicted"]]
        .sum()
        .assign(observed=lambda x: x["observed"])
        .assign(predicted=lambda x: x["predicted"])
    )
    ax = df_.loc[:, ["observed", "predicted"]].plot(style=".", ax=ax)
    ax.set(
        ylabel=y_label if y_label is not None else None,
        title=title if title is not None else "Train: Observed vs Predicted",
    )
```

```python
fig, ax = plt.subplots(ncols=1, figsize=(15, 5))
plot_obs_pred(
    df=updated_features_testset,
    feature=target[0],
    observed=target[0],
    predicted=gamma_regressor.predict(updated_features_testset[train_variables]),
    y_label="Average claim severity",
    title="Predicted vs Observed",
    ax=ax
)
```

**What to look for**:
- Points should cluster along the diagonal (predicted ≈ observed)
- Systematic deviations indicate model bias
- Scatter around the diagonal indicates prediction uncertainty

---

## 9. Conclusion

### Key Takeaways

1. **What We Built**: Two severity models predicting insurance claim amounts

2. **The Models**:
   - Baseline: Simple mean prediction (benchmark)
   - Gamma Regressor: GLM designed for positive continuous data

3. **Critical Insight**: Severity modelling uses only policies with claims (y > 0), unlike frequency modelling which uses all policies

4. **Best Practice**: Gamma regression is theoretically appropriate for claim amounts due to:
   - Strictly positive values
   - Right-skewed distribution
   - Log link ensuring positive predictions

5. **Next Steps**:
   - Combine frequency × severity for pure premium calculation
   - Explore feature importance for pricing factor identification
   - Consider additional features to improve prediction accuracy

### Questions to Explore

1. Why do we exclude zero-claim policies when modelling severity?
2. How does the choice of regularization parameter (alpha) affect the Gamma model?
3. What features are most predictive of high-severity claims?
4. How would you handle extreme outliers in claim amounts?

---

## Appendix: Technical Reference

### File Locations

```
Data:
  - data/input/Motor_vehicle_insurance_data.csv
  - data/input/sample_type_claim.csv

Feature Engineering:
  - src/model/freq_sev/feature.py

Notebook:
  - notebook/freq_sev/severity.ipynb
```

### Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Severity** | Monetary amount of a claim when it occurs |
| **Gamma Distribution** | Probability distribution for positive continuous data |
| **Right-skewed** | Distribution with a long tail of high values |
| **Log Link** | Ensures predictions are always positive |
| **Regularization** | Technique to prevent overfitting (alpha parameter) |

### Comparison with Frequency Modelling

| Aspect | Frequency | Severity |
|--------|-----------|----------|
| Question answered | "How many claims?" | "How much per claim?" |
| Target variable | claims_frequency | Cost_claims_year |
| Data subset | All policies | Claims > 0 only |
| Model type | Poisson Regressor | Gamma Regressor |
| Distribution | Discrete counts | Positive continuous |

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**For Questions**: Refer to the severity.ipynb notebook for the complete implementation
