# Motor Vehicle Insurance Data Exploration: Understanding Claims Frequency and Severity

## 1. Introduction

Welcome! This guide walks you through the **exploratory data analysis (EDA)** process for our motor vehicle insurance dataset. Think of EDA as being a detective, we're investigating the data to uncover patterns, spot anomalies, and develop hypotheses before building predictive models.

### What is Exploratory Data Analysis?

EDA is the critical first step in any data science project. Before jumping into model building, we need to understand:
- What does our data look like?
- Are there patterns we can leverage?
- Are there issues we need to address?

### Objectives of This Exploration

Our exploration aims to answer these key business questions:

1. **Claims Frequency Distribution**: How many claims do policyholders typically make? Is the distribution skewed?
2. **Claims Severity Patterns**: When claims do occur, how costly are they? What's the range?

### Why This Matters for Underwriting

In insurance, one way to calculate **pure premium** (expected claim cost) is calculated as:

```
Pure Premium = Frequency × Severity
```

Understanding both components separately helps us:
- Price policies more accurately
- Identify high-risk segments
- Develop targeted risk mitigation strategies
---

## 2. Data Overview

### Loading the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure pandas to show all data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load both datasets
motor_df = pd.read_csv('../../data/input/exp/Motor_vehicle_insurance_data.csv', delimiter=";")
claims_df = pd.read_csv('../../data/input/exp/sample_type_claim.csv', delimiter=';')
```

---

## 3. Exploratory Data Analysis (EDA) Techniques

### 3.1 Creating Analysis-Ready Features

Before diving into analysis, we created two important derived features:

#### Feature 1: Claims Bin (Categorizing Claim Counts)

**Purpose**: Group policyholders by their number of claims for frequency analysis.

```python
def assign_claims_bin(claims_value):
    if claims_value <= 25:
        return str(claims_value)
    else:
        return '25+'

# Apply to each row
motor_df['claim_bin'] = motor_df['N_claims_year'].apply(assign_claims_bin)
```

**Why 25+?**: Looking at the data, very few policyholders have more than 25 claims in a year. Grouping these together prevents sparse categories from skewing our analysis.

#### Feature 2: Unique Policy ID

**Purpose**: Create a truly unique identifier for each policy-year combination.

```python
# Combine ID with renewal dates to create unique policy identifier
motor_df['unique_policy_id'] = (
    motor_df['ID'].astype(str) + 
    motor_df['Date_last_renewal'].astype(str) + 
    motor_df['Date_next_renewal'].astype(str)
)
```

**Why This Matters**: The same customer (ID) can have multiple policy records across different years. By combining ID with renewal dates, we ensure each policy-year is uniquely identified.

**Example**:
```
ID=1, Date_last_renewal=05/11/2015, Date_next_renewal=05/11/2016 
→ unique_policy_id = "105/11/201505/11/2016"

ID=1, Date_last_renewal=05/11/2016, Date_next_renewal=05/11/2017 
→ unique_policy_id = "105/11/201605/11/2017"
```

### 3.2 Claims Frequency Distribution Analysis

This is where we answer: **"How many claims do policyholders typically make?"**

#### Method: GroupBy Aggregation

```python
grouping = (
    motor_df
    .groupby('claim_bin', observed=True)
    .agg({
        'unique_policy_id': 'nunique',  # Count unique policies
        'N_claims_year': 'sum'          # Sum total claims
    })
    .rename(columns={
        'unique_policy_id': 'Policies', 
        'N_claims_year': 'Claims'
    })
    # Sort numerically (convert string to int for proper sorting)
    .sort_values(by=['claim_bin'], key=lambda x: x.astype(int), ascending=True)
    .reset_index()
)

# Calculate proportion of total policies
grouping['Proportion'] = (grouping['Policies'] / grouping['Policies'].sum()).round(4)
```

**What's Happening Step-by-Step**:

1. **`groupby('claim_bin')`**: Groups all rows by their claims category (0, 1, 2, ..., 25+)
2. **`agg({'unique_policy_id': 'nunique'})`**: Counts unique policies in each group
3. **`agg({'N_claims_year': 'sum'})`**: Sums all claims in each group
4. **`sort_values(..., key=lambda x: x.astype(int))`**: Sorts numerically (important because '10' would come before '2' alphabetically!)
5. **Proportion calculation**: Shows what percentage of policies fall into each category

### 3.3 Claims Severity Distribution Analysis

This answers: **"When claims occur, how much do they cost?"**

```python
claims_summary = (
    claims_df[['Cost_claims_year']]
    .drop_duplicates(keep='first')  # Each policyholder's total claim cost
    .describe()                      # Summary statistics
    .drop('count')                   # Remove count row
    .T                               # Transpose for readability
    .reset_index(drop=True)
)
```

**Key Insight**: The claims data already has aggregated costs per policyholder. We use `drop_duplicates()` to get unique claim amounts for severity analysis.

### 3.4 Visualization Techniques

We created two complementary visualizations to understand the severity distribution:

```python
claims = claims_df[['Cost_claims_year']].drop_duplicates(keep='first')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original scale histogram
ax1.hist(claims['Cost_claims_year'])
ax1.set_xlabel("Average Claims")
ax1.set_title("Claims Distribution (Original Scale)")

# Logarithmic scale histogram
ax2.hist(np.log(claims['Cost_claims_year']))
ax2.set_xlabel("Logarithmic Average Claims")
ax2.set_title("Claims Distribution (Log Scale)")

plt.tight_layout()
plt.show()
```

**Why Two Histograms?**

1. **Original Scale**: Shows the actual distribution, but extreme values (outliers) can compress most of the data into a small visual range
2. **Logarithmic Scale**: Transforms the data to reveal patterns in the lower range while still showing the full distribution

This is a common technique when dealing with **right-skewed** financial data like claim costs.

---

## 4. Findings and Insights

### 4.1 Claims Frequency Distribution

The frequency analysis revealed a critical insight for insurance pricing:

| Claims | Policies | Total Claims | Proportion |
|--------|----------|--------------|------------|
| 0 | 85,909 | 0 | **81.39%** |
| 1 | 9,539 | 9,539 | 9.04% |
| 2 | 4,961 | 9,922 | 4.70% |
| 3 | 2,435 | 7,305 | 2.31% |
| 4 | 1,190 | 4,760 | 1.13% |
| 5 | 609 | 3,045 | 0.58% |
| 6 | 318 | 1,908 | 0.30% |
| 7 | 227 | 1,589 | 0.22% |
| 8 | 136 | 1,088 | 0.13% |
| 9 | 82 | 738 | 0.08% |
| 10 | 61 | 610 | 0.06% |
| 11-25+ | <100 each | Varies | <0.02% each |
| **Total** | **105,555** | **41,662** | **100%** |

#### Key Insights from Frequency Analysis

1. **Zero-Inflation**: A staggering **81.39%** of policies have zero claims in a given year. This is typical in insurance and suggests we may need specialized models (like Zero-Inflated Poisson) for frequency prediction.

2. **Rapid Decay**: The proportion drops dramatically:
   - 0 claims: 81.39%
   - 1 claim: 9.04%
   - 2 claims: 4.70%
   - 3+ claims: <5% combined

3. **Long Tail**: While rare, some policyholders have up to 25 claims in a year (and potentially more). These high-frequency claimants require special attention.

4. **Poisson-Like Distribution**: The pattern suggests a Poisson distribution might be appropriate for modeling.

### 4.2 Claims Severity Distribution

When claims DO occur, here's what they look like:

| Statistic | Value |
|-----------|-------|
| **Mean** | €1,041.27 |
| **Standard Deviation** | €4,930.44 |
| **Minimum** | €40.05 |
| **25th Percentile** | €150.40 |
| **Median (50th)** | €337.67 |
| **75th Percentile** | €758.33 |
| **Maximum** | €236,285.18 |

#### Key Insights from Severity Analysis

1. **High Variance**: The standard deviation (€4,930) is nearly 5x the mean (€1,041), indicating extreme variability in claim costs.

2. **Right-Skewed Distribution**: 
   - Mean (€1,041) >> Median (€338)
   - This indicates a few very large claims pulling the mean up
   - Most claims are relatively small

3. **Extreme Outliers**: The maximum claim of €236,285 is 227x the mean! These catastrophic claims significantly impact the overall risk.

4. **Typical Claims Are Small**: 
   - 50% of claims are under €338
   - 75% of claims are under €758
   - But the remaining 25% drive most of the cost

#### Visual Representation

The histogram visualizations reveal:

**Original Scale Histogram**:
- Heavily right-skewed
- Most claims clustered near €0-€2,000
- Long tail extending to €236,285
- Hard to see detail in the common range

**Logarithmic Scale Histogram**:
- More symmetric, approximately normal
- Reveals the underlying distribution pattern
- Shows that log-transformed severity might be easier to model
- Suggests Gamma or Log-Normal distribution for severity modeling

## 5. Conclusion

### Summary of Key Findings

1. **Claims Frequency**:
   - 81% of policies have zero claims (zero-inflation)
   - Distribution follows a Poisson-like pattern with long tail
   - Suggests need for Zero-Inflated or Negative Binomial models

2. **Claims Severity**:
   - Highly right-skewed with extreme outliers
   - Mean (€1,041) much higher than median (€338)
   - Log transformation normalizes the distribution
   - Suggests Gamma or Log-Normal models for severity

3. **Business Implications**:
   - Base premium can be conservative (most policies won't claim)
   - Need to account for high-severity outliers in pricing
   - Risk segmentation is essential for accurate pricing

### What We Learned

- **EDA is Essential**: Before building any model, understanding your data's shape, distribution, and quirks is crucial
- **Insurance Data is Unique**: Zero-inflation and right-skewed severity are not uncommon
- **Visualization Matters**: The log-scale histogram revealed patterns invisible in the original scale

---

## Appendix: Technical Reference

### File Locations

```
Raw Data:
  - data/Motor_vehicle_insurance_data.csv
  - data/sample_type_claim.csv

Exploration Notebook:
  - notebook/freq-sev-approach/explore.ipynb

Dataset Preparation:
  - notebook/freq-sev-approach/dataset.ipynb
```

### Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Key Functions Used

| Function | Purpose | Example |
|----------|---------|---------|
| `pd.read_csv()` | Load CSV data | `pd.read_csv(path, delimiter=';')` |
| `df.apply()` | Apply function to column | `df['col'].apply(func)` |
| `df.groupby()` | Group data for aggregation | `df.groupby('col').agg({...})` |
| `df.describe()` | Summary statistics | `df[['col']].describe()` |
| `df.drop_duplicates()` | Remove duplicate rows | `df.drop_duplicates(keep='first')` |
| `plt.hist()` | Create histogram | `plt.hist(data)` |
| `np.log()` | Natural logarithm | `np.log(df['col'])` |

### Recommended Reading
- **Insurance Fundamentals**: [insurance-a-mind-dump-so-you-get-a-head-start](https://medium.com/@olumideodetunde/insurance-a-mind-dump-so-you-get-a-head-start-e1b8dbc4ce76)

---
**Document Version**: 1.0  
**Last Updated**: January 2026  
**Source Notebook**: `notebook/freq-sev-approach/explore.ipynb`  
**For Questions**: Refer to the exploration notebook for complete implementation
