# Frequency Modelling: Predicting Insurance Claim Counts

## 1. Introduction

Welcome! This guide walks you through how we build **frequency models** to predict how many claims a policyholder will make. This is a critical component of insurance pricing.

### What is Frequency Modelling?

Frequency modelling predicts the **number of claims** a policyholder will file during a policy period. In insurance mathematics:

```
Premium = Frequency × Severity
```

Where:
- **Frequency**: How many claims (what we're modelling here)
- **Severity**: How much each claim costs (covered separately)

### Purpose

By accurately predicting claim frequency, insurers can:
- Price policies fairly based on risk
- Identify high-risk customer segments
- Allocate capital reserves appropriately
- Design targeted risk mitigation strategies

### Real-World Context

Imagine pricing a motor insurance policy. Two customers with identical vehicles might have very different claim likelihoods based on:
- Where they live (urban vs rural)
- Their driving experience
- The type of vehicle usage

Frequency models capture these relationships mathematically.

---

## 2. Data Preparation

### Step 1: Load Required Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
```

### Step 2: Create the Dataset

We load two data sources and merge them to create our modelling dataset:

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

**What just happened?**
- Loaded insurance policy data and claims history
- Counted claims per policyholder per year
- Merged datasets, filling missing claims with 0 (no claims)

### Step 3: Split into Training and Test Sets

```python
trainset, testset = train_test_split(
    dataset, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    shuffle=True        # Randomize the split
)
```

**Why split the data?**
- **Training set (80%)**: Used to fit the model
- **Test set (20%)**: Used to evaluate performance on unseen data
- Prevents overfitting and gives realistic performance estimates

---

## 3. Feature Engineering

Features are created using a dedicated feature engineering module:

```python
from src.model.freq_sev.feature import main as feature_main

features_trainset = feature_main(trainset)
features_testset = feature_main(testset)
```

The feature engineering process transforms raw data into predictive features suitable for modelling. See the feature engineering documentation for details.

---

## 4. Understanding the Target Variable

Before modelling, we examine the distribution of our target variable `claims_frequency` on the **training set only**. This is important to avoid data leakage from the test set influencing our modelling decisions.

### Distribution Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize the distribution
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_title('Claims Frequency Distribution')
features_trainset['claims_frequency'].hist(bins=4, log=True, ax=ax)

# Summary statistics
print(f"Average claims frequency: {np.average(features_trainset['claims_frequency'])}")
print(f"Fraction with zero claims: {(features_trainset['claims_frequency']==0).mean():.2%}")
```

**Key Findings**:
- **Average claims frequency**: ~0.07 (very low)
- **Zero-claim proportion**: ~95% of policyholders file no claims

### Checking Poisson Assumptions

For Poisson regression, we need the **mean ≈ variance** (equidispersion):

```python
mean_claims = features_trainset['claims_frequency'].mean()
var_claims = features_trainset['claims_frequency'].var()
print(f"Mean: {mean_claims:.4f}")
print(f"Variance: {var_claims:.4f}")
```

**Result**: Mean (0.0703) and Variance (0.1214) are reasonably close, supporting the use of Poisson regression.

**Interpretation**: The distribution is:
- Unimodal (single peak at zero)
- Right-skewed (most values are 0, with a long tail)
- Suitable for count-based models like Poisson regression

---

## 5. Model Development

We develop and compare four models of increasing complexity.

### Selected Features

```python
training_variables = [
    'Car_age_years',         # Vehicle age
    'Type_risk',             # Vehicle usage type
    'Area',                  # Geographic location
    'Value_vehicle',         # Vehicle monetary value
    'Distribution_channel',  # How policy was purchased
    'Cylinder_capacity'      # Engine size
]
target = ['claims_frequency']
```

### Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

def model_evaluation_metrics(estimator, df_test, target_variable, training_variables):
    y_pred = estimator.predict(df_test[training_variables])
    
    print(f"MSE: {mean_squared_error(df_test[target_variable], y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(df_test[target_variable], y_pred):.3f}")
    
    # Handle non-positive predictions for Poisson deviance
    mask = y_pred > 0
    if (~mask).any():
        print(f"WARNING: {(~mask).sum()} invalid predictions ignored")
    
    print(f"Mean Poisson Deviance: {mean_poisson_deviance(df_test[target_variable][mask], y_pred[mask]):.3f}")
```

**Why these metrics?**
- **MSE (Mean Squared Error)**: Penalizes large errors heavily
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **Poisson Deviance**: Appropriate for count data; measures goodness-of-fit for Poisson models

---

## 6. Model 1: Baseline (Mean Prediction)

The simplest model: predict the average claim frequency for everyone.

```python
from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor(strategy="mean")
baseline_model = dummy_regressor.fit(
    features_trainset[training_variables], 
    features_trainset[target]
)
```

**Results**:
| Metric | Value |
|---|---|
| MSE | 0.120 |
| MAE | 0.131 |
| Poisson Deviance | 0.428 |

**Interpretation**: This is our benchmark. Any useful model should beat these numbers.

---

## 7. Model 2: Ridge Regression

Ridge regression is a linear model with L2 regularization to prevent overfitting.

```python
from sklearn.linear_model import Ridge

ridge_glm = Ridge(alpha=1)
ridge_model = ridge_glm.fit(
    features_trainset[training_variables], 
    features_trainset[target]
)
```

**Results**:
| Metric | Value |
|---|---|
| MSE | 0.119 |
| MAE | 0.131 |
| Poisson Deviance | 0.424 |

**Interpretation**: Slight improvement over baseline. Ridge captures some linear relationships between features and claim frequency.

---

## 8. Model 3: Poisson Regression

Poisson regression is specifically designed for count data like claim frequencies.

```python
from sklearn.linear_model import PoissonRegressor

poisson_regressor = PoissonRegressor(
    alpha=1e-12,           # Minimal regularization
    solver='newton-cholesky',
    max_iter=300
)
poisson_model = poisson_regressor.fit(
    features_trainset[training_variables], 
    features_trainset[target].values.ravel()
)
```

**Results**:
| Metric | Value |
|---|---|
| MSE | 0.119 |
| MAE | 0.131 |
| Poisson Deviance | 0.424 |

**Why Poisson Regression?**
- Assumes target follows a Poisson distribution (appropriate for counts)
- Naturally handles non-negative predictions
- Log link function captures multiplicative effects

**Interpretation**: Performance similar to Ridge, but theoretically more appropriate for count data.

---

## 9. Model 4: Gradient Boosting Machine (GBM)

GBM is an ensemble method that can capture non-linear relationships.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

gbm_regressor = HistGradientBoostingRegressor(
    loss='poisson',       # Poisson loss for count data
    max_leaf_nodes=128    # Control tree complexity
)
gbm_model = gbm_regressor.fit(
    features_trainset[training_variables], 
    features_trainset['claims_frequency']
)
```

**Results**:
| Metric | Value |
|---|---|
| MSE | 0.120 |
| MAE | 0.131 |
| Poisson Deviance | 0.426 |

**Why GBM?**
- Can capture non-linear relationships
- Handles feature interactions automatically
- Often achieves state-of-the-art performance

**Interpretation**: Surprisingly, GBM doesn't outperform simpler models here. This suggests:
- The feature-target relationships may be mostly linear
- More features or feature engineering might be needed
- The data may have limited predictive signal

---

## 10. Model Comparison

### Summary Table

| Model | MSE | MAE | Poisson Deviance |
|---|---|---|---|
| Baseline (Mean) | 0.120 | 0.131 | 0.428 |
| Ridge Regression | 0.119 | 0.131 | 0.424 |
| Poisson Regression | 0.119 | 0.131 | 0.424 |
| Gradient Boosting | 0.120 | 0.131 | 0.426 |

### Visual Comparison

The models are compared by examining the distribution of predicted values against actual values for both training and test sets.

**Key Observations**:
1. All models produce similar prediction distributions
2. Predictions are concentrated in a narrow range (reflecting low base rate)
3. The "spike at zero" in actual data is smoothed into a continuous distribution by all models

### Recommendations

For this dataset:
1. **Poisson Regression** is recommended due to:
   - Theoretical appropriateness for count data
   - Interpretable coefficients
   - Comparable performance to complex models

2. **For production**, consider:
   - Additional feature engineering
   - Zero-inflated models (given 95% zero claims)
   - Ensemble approaches combining multiple models

---

## 11. Conclusion

### Key Takeaways

1. **What We Built**: Four frequency models predicting insurance claim counts

2. **The Models**:
   - Baseline: Simple mean prediction (benchmark)
   - Ridge: Linear model with regularization
   - Poisson: Count-specific regression model
   - GBM: Non-linear ensemble method

3. **Key Finding**: All models perform similarly, suggesting:
   - The selected features have limited predictive power
   - Relationships may be inherently difficult to predict
   - Consider additional features or external data sources

4. **Best Practice**: Poisson regression is theoretically appropriate for count data and provides interpretable results

5. **Next Steps**:
   - Build severity models (predicting claim costs)
   - Combine frequency × severity for premium calculation
   - Explore variable importance for rating factor identification
   - Consider zero-inflated models for the high proportion of zero claims

### Questions to Explore

1. Why might GBM not outperform linear models here?
2. How would you handle the 95% zero-claim observations differently?
3. What additional features might improve prediction accuracy?
4. How would you deploy these models in a production pricing system?

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
  - notebook/freq_sev/frequency.ipynb
```

### Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
```

### Key Concepts

| Concept | Description |
|---|---|
| **Frequency** | Number of claims per policy period |
| **Poisson Distribution** | Probability distribution for count data |
| **Equidispersion** | Mean equals variance (Poisson assumption) |
| **Poisson Deviance** | Goodness-of-fit measure for count models |
| **Regularization** | Technique to prevent overfitting |

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**For Questions**: Refer to the frequency.ipynb notebook for the complete implementation
