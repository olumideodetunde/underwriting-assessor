# Motor Vehicle Insurance Data Exploration: Understanding Claims Frequency and Severity

## 1. Introduction

Welcome! This guide walks you through the **exploratory data analysis (EDA)** process for our motor vehicle insurance dataset. Think of EDA as being a detective — we're investigating the data to uncover patterns, spot anomalies, and develop hypotheses before building predictive models.

### What is Exploratory Data Analysis?

EDA is the critical first step in any data science project. Before jumping into model building, we need to understand:
- What does our data look like?
- Are there patterns we can leverage?
- Are there issues we need to address?

### Objectives of This Exploration

Our exploration aims to answer these key business questions:

1. **Claims Frequency Distribution**: How many claims do policyholders typically make? Is the distribution skewed?
2. **Claims Severity Patterns**: When claims do occur, how costly are they? What's the range?
3. **Policy Segmentation**: Can we identify natural groupings in the data that might inform pricing strategies?

### Why This Matters for Underwriting

In insurance, the **pure premium** (expected claim cost) is calculated as:

```
Pure Premium = Frequency × Severity
```

Understanding both components separately helps us:
- Price policies more accurately
- Identify high-risk segments
- Develop targeted risk mitigation strategies

---

## 2. Data Overview

### Datasets Explored

We work with two interconnected datasets:

#### Dataset 1: Motor Vehicle Insurance Data
**File**: `data/input/exp/Motor_vehicle_insurance_data.csv`

| Characteristic | Details |
|----------------|---------|
| **Records** | 105,555 policy-year observations |
| **Columns** | 30 features |
| **Time Period** | November 2015 – December 2019 |
| **Delimiter** | Semicolon (`;`) |

#### Dataset 2: Claims Data
**File**: `data/input/exp/sample_type_claim.csv`

| Characteristic | Details |
|----------------|---------|
| **Records** | Claims broken down by type |
| **Key Columns** | ID, Cost_claims_year, Cost_claims_by_type |
| **Delimiter** | Semicolon (`;`) |

### Key Variables in the Insurance Data

The dataset contains rich information across several categories:

| Category | Variables | Description |
|----------|-----------|-------------|
| **Identifiers** | ID, unique_policy_id | Policy and customer identifiers |
| **Dates** | Date_start_contract, Date_last_renewal, Date_next_renewal, Date_birth, Date_driving_licence | Temporal information |
| **Customer Info** | Seniority, Distribution_channel, Policies_in_force | Customer relationship data |
| **Vehicle** | Year_matriculation, Power, Cylinder_capacity, Value_vehicle, N_doors, Type_fuel, Length, Weight | Vehicle characteristics |
| **Risk Factors** | Type_risk, Area, Second_driver | Risk classification variables |
| **Claims** | N_claims_year, Cost_claims_year, N_claims_history, R_Claims_history | Claims history |
| **Policy** | Premium, Payment, Lapse, Date_lapse | Policy status and payment |

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

**Pro Tip**: Always check your delimiter! European datasets often use semicolons (`;`) instead of commas (`,`). Loading with the wrong delimiter will result in a single column containing all your data.

---

## 3. Exploratory Data Analysis (EDA) Techniques

### 3.1 Creating Analysis-Ready Features

Before diving into analysis, we created two important derived features:

#### Feature 1: Claims Bin (Categorizing Claim Counts)

**Purpose**: Group policyholders by their number of claims for frequency analysis.

```python
def assign_claims_bin(claims_value):
    """
    Categorize number of claims into bins.
    Claims 0-25 get their own category, 25+ grouped together.
    """
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

4. **Poisson-Like Distribution**: The pattern suggests a Poisson or Negative Binomial distribution might be appropriate for modeling.

#### Visual Representation

```
Claims Frequency Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0 claims  ████████████████████████████████████████████████████ 81.39%
1 claim   █████ 9.04%
2 claims  ███ 4.70%
3 claims  █ 2.31%
4+ claims █ <2.5%
```

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

```
Claims Severity Distribution (Conceptual)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original Scale:
€0-500     ████████████████████████████████████████ Most claims here
€500-1K    ████████
€1K-5K     ████
€5K-50K    █
€50K+      ▏ Rare but impactful

Log Scale:
~5 (€150)  ████████
~6 (€400)  ████████████████████████████████████████ Peak
~7 (€1.1K) ████████████████████
~8 (€3K)   ████████
~10+(€22K+)██
```

### 4.3 Combined Insights for Premium Calculation

Combining frequency and severity:

| Metric | Value | Implication |
|--------|-------|-------------|
| Claims per Policy | 0.395 (41,662 / 105,555) | On average, ~40% of a claim per policy-year |
| Average Severity | €1,041.27 | When claims happen, average cost |
| **Estimated Pure Premium** | **~€411** | 0.395 × €1,041 |

**Important Caveat**: This is a simplified calculation. Actual premium calculation must account for:
- Expenses and profit margin
- Risk segmentation by customer/vehicle characteristics
- Regulatory requirements
- Market competition

---

## 5. Challenges Encountered

### Challenge 1: Data Delimiter Issues

**Problem**: The CSV files use semicolons (`;`) instead of commas (`,`).

**Solution**: Explicitly specify `delimiter=";"` in `pd.read_csv()`.

**Learning Point**: Always inspect your raw data file before loading. A quick `head` command or opening in a text editor can save hours of debugging.

```python
# Wrong - would create a single column with all data
df = pd.read_csv('file.csv')  

# Correct - properly parses the columns
df = pd.read_csv('file.csv', delimiter=';')
```

### Challenge 2: Missing Values

**Problem**: Several columns have missing values:
- `Length`: ~20% missing (vehicle length not recorded for all vehicles)
- `Date_lapse`: Mostly NaN (only populated when policy lapses)

**Solution**: 
- For `Length`: Will need imputation or exclusion in modeling
- For `Date_lapse`: NaN is meaningful (policy is still active)

**Learning Point**: Missing values aren't always "bad data" — sometimes they carry information!

### Challenge 3: Sorting String Numbers

**Problem**: When sorting claim bins, "10" comes before "2" alphabetically.

**Solution**: Use a key function to convert strings to integers for sorting:

```python
.sort_values(by=['claim_bin'], key=lambda x: x.astype(int), ascending=True)
```

### Challenge 4: Multiple Records per Customer

**Problem**: The same customer ID appears multiple times (one per policy year).

**Solution**: Created `unique_policy_id` by combining ID with renewal dates.

**Learning Point**: Always understand your data's granularity. Is each row a customer? A policy? A policy-year? This affects all subsequent analysis.

---

## 6. Conclusion

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
- **Insurance Data is Unique**: Zero-inflation and right-skewed severity are hallmarks of insurance data
- **Visualization Matters**: The log-scale histogram revealed patterns invisible in the original scale

---

## 7. Next Steps

Based on our exploration, here are the recommended next steps:

### 7.1 Feature Engineering

1. **Age-Related Features**:
   - Calculate driver age from `Date_birth`
   - Calculate driving experience from `Date_driving_licence`
   - Create age bands (young driver, experienced, senior)

2. **Vehicle Age Features**:
   - Calculate vehicle age from `Year_matriculation`
   - Create depreciated value estimate

3. **Customer Relationship Features**:
   - Tenure bands from `Seniority`
   - Multi-policy indicator from `Policies_in_force`

4. **Exposure Calculation**:
   - Calculate policy exposure (fraction of year covered)
   - Essential for proper frequency modeling

### 7.2 Model Development

1. **Frequency Model Options**:
   - Poisson Regression (baseline)
   - Negative Binomial (for overdispersion)
   - Zero-Inflated Poisson/NB (for excess zeros)

2. **Severity Model Options**:
   - Gamma GLM (for positive continuous data)
   - Log-Normal Regression
   - Tweedie Distribution (combined frequency-severity)

### 7.3 Further Analysis

1. **Correlation Analysis**: Examine relationships between features and claims
2. **Segmentation Analysis**: Identify high-risk customer/vehicle profiles
3. **Temporal Patterns**: Check for seasonality or trends in claims
4. **Geographic Analysis**: Explore `Area` variable impact on claims

### 7.4 Data Quality Improvements

1. **Handle Missing `Length` Values**: Impute using vehicle type/model
2. **Validate Consistency**: Ensure vehicle characteristics are consistent across years
3. **Outlier Treatment**: Develop strategy for extreme severity values

---

## Appendix: Technical Reference

### File Locations

```
Raw Data:
  - data/input/exp/Motor_vehicle_insurance_data.csv
  - data/input/exp/sample_type_claim.csv

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

- **Zero-Inflated Models**: Understanding when and why to use them
- **GLMs in Insurance**: Generalized Linear Models for pricing
- **Log Transformations**: When and how to apply them
- **Insurance Pricing Fundamentals**: [insurance-a-mind-dump-so-you-get-a-head-start](https://medium.com/@olumideodetunde/insurance-a-mind-dump-so-you-get-a-head-start-e1b8dbc4ce76)

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Source Notebook**: `notebook/freq-sev-approach/explore.ipynb`  
**For Questions**: Refer to the exploration notebook for complete implementation
