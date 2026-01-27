# Feature Engineering: Variables Available at Initial Quote Request

## 1. Introduction

Welcome! This guide walks you through the **feature engineering** process for our motor vehicle insurance dataset. Think of feature engineering as crafting the raw materials: we're transforming raw data into meaningful inputs that help our models make better predictions.

### What is Feature Engineering?

Feature engineering is the process of using domain knowledge to create new variables (features) from raw data. These engineered features often capture relationships that raw data cannot express directly.

### Objectives of This Notebook

Our feature engineering aims to:

1. **Prepare the dataset**: Merge insurance and claims data into a single analysis-ready dataset
2. **Research relevant features**: Identify which risk factors matter for motor insurance pricing
3. **Engineer new features**: Create derived variables like driver age, driving experience, and power-to-weight ratio
4. **Explore the features**: Understand distributions and relationships before modeling

### Why This Matters for Underwriting

In insurance, features available at the **initial quote request** are crucial because:
1. They're the only information available when pricing a new policy
2. They must predict future claims without historical customer data
3. Good features lead to accurate risk segmentation and fair pricing

---

## 2. Data Overview

### Input Variables

The variables we're working with are obtained by underwriters at first contact:

| Variable | Description |
|----------|-------------|
| `ID` | Unique identifier for each policyholder |
| `Distribution_channel` | Channel through which policy was contracted (0=Agent, 1=Broker) |
| `Date_birth` | Date of birth of the insured (DD/MM/YYYY) |
| `Date_driving_licence` | Date of driver's license issuance (DD/MM/YYYY) |
| `Premium` | Net premium amount for the current year |
| `Type_risk` | Risk type (1=Motorbikes, 2=Vans, 3=Passenger cars, 4=Agricultural) |
| `Area` | Location type (0=Rural, 1=Urban with >30,000 inhabitants) |
| `Second_driver` | Multiple drivers declared (0=No, 1=Yes) |
| `Year_matriculation` | Year of vehicle registration (YYYY) |
| `Power` | Vehicle power in horsepower |
| `Cylinder_capacity` | Cylinder capacity of the vehicle |
| `Value_vehicle` | Market value of vehicle on 31/12/2019 |
| `N_doors` | Number of vehicle doors |
| `Type_fuel` | Fuel type (P=Petrol, D=Diesel) |
| `Length` | Length of vehicle in meters |
| `Weight` | Weight of vehicle in kilograms |

### Loading the Data

```python
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

insurance_initiation_variables_path = "../../data/output/Insurance_Initiation_Variables.csv"
claims_variables_path = "../../data/input/exp/sample_type_claim.csv"

insurance_df = pd.read_csv(insurance_initiation_variables_path, delimiter=';')
claims_df = pd.read_csv(claims_variables_path, delimiter=';')
```

---

## 3. Dataset Preparation

Before engineering features, we need to prepare a unified dataset by merging insurance data with claims information.

### 3.1 Aggregate Claims by Policyholder

**Purpose**: Sum all claims for each policyholder-year combination.

```python
claim_grouping_columns = ['ID', 'Cost_claims_year']
claim_aggregation_column = 'Cost_claims_by_type'

claims_aggregated = (
    claims_df
    .groupby(claim_grouping_columns, as_index=False)[claim_aggregation_column]
    .sum()
)
```
**Why Aggregate?**: A single policyholder may have multiple claim types in a year. Summing them gives us total claim cost per policy-year.

### 3.2 Merge Insurance and Claims Data

**Purpose**: Combine policyholder features with their claim history.

```python
merging_columns = ['ID', 'Cost_claims_year']
dataset = insurance_df.merge(claims_aggregated, on=merging_columns, how='left')
dataset[claim_aggregation_column] = dataset[claim_aggregation_column].fillna(0)
dataset['claims_frequency'] = (dataset[claim_aggregation_column] > 0).astype(int)
```

**What's Happening**:
1. **Left join**: Keep all insurance records, even those with no claims
2. **Fill NaN → 0**: Policyholders with no claims get a claim cost of 0
3. **Create binary target**: `claims_frequency` = 1 if any claim exists, 0 otherwise

### 3.3 Train-Test Split

**Purpose**: Reserve data for model evaluation.

```python
test_ratio = 0.2
to_shuffle = False
if to_shuffle:
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(dataset) * (1 - test_ratio))
trainset = dataset.iloc[:split_index].reset_index(drop=True)
testset = dataset.iloc[split_index:].reset_index(drop=True)
```

**Why No Shuffle?**: Preserving temporal ordering can be important for insurance data where policies evolve over time.

---

## 4. Research: Baseline Risk Factors

Before engineering features, we research what factors actually matter for motor insurance risk. Based on domain knowledge from [this research paper](https://www.researchgate.net/publication/338007809_An_Analysis_of_the_Risk_Factors_Determining_Motor_Insurance_Premium_in_a_Small_Island_State_The_Case_of_Malta), here's our feature inventory:

| Risk Factor | Status | Approach |
|-------------|--------|----------|
| Type of Vehicle | Exists | Use `Type_risk` directly |
| Value of Vehicle | Exists | Use `Value_vehicle` directly |
| Age of Driver | To Engineer | Calculate from `Date_birth` |
| Vehicle Technology | To Engineer | Proxy via `Year_matriculation` (newer = better tech) |
| Geographic Location | Exists | Use `Area` directly |
| Repair Cost Proxy | Exists | Use `Value_vehicle` as proxy |
| Power to Weight Ratio | To Engineer | Calculate `Power / Weight` |
| Occupation | Missing | Not captured in dataset |
| Medical Condition | Missing | Not captured in dataset |

**Key Insight**: While telematics data is increasingly used in modern insurance, traditional features still form the foundation of risk assessment.

---

## 5. Feature Implementation

### 5.1 Features to Engineer

| Feature | Formula | Unit | Rationale |
|---------|---------|------|-----------|
| `Driver_age_years` | today - Date_birth | Years | Younger drivers = higher risk |
| `Driver_experience_years` | today - Date_driving_licence | Years | Less experience = higher risk |
| `Car_age_years` | today_year - Year_matriculation | Years | Older cars = less safety tech |
| `power_to_weight` | Power / Weight | HP/kg | Higher ratio = sportier = riskier |

### 5.2 Helper Functions

```python
def convert_to_datetime(value:object, format:str="%d/%m/%Y", yearfirst:bool=True) -> Any:
    return pd.to_datetime(arg=value, format=format, yearfirst=yearfirst)

def take_datetime_difference_in_years(first_datetime:datetime, second_datetime:datetime, interval) -> float:
    diff = (second_datetime - first_datetime) / np.timedelta64(1, interval)
    diff_years = diff/365.25
    return diff_years

def take_int_difference(first_number:int, second_number:int) -> int:
    return abs(first_number - second_number)
```

**Why 365.25?**: Accounts for leap years when converting days to years.

### 5.3 Apply Transformations

```python
today_date = pd.Timestamp.today()
today_year = today_date.year

features_trainset = (
    trainset
    .assign(
        Date_birth_dt=trainset['Date_birth'].apply(convert_to_datetime),
        Date_driving_licence_dt=trainset['Date_driving_licence'].apply(convert_to_datetime),
        power_to_weight = trainset['Power'] / trainset['Weight'],
        Car_age_years= trainset['Year_matriculation'].apply(take_int_difference, args=(today_year,))
    )
    .assign(
        Driver_age_years=lambda df: df['Date_birth_dt'].apply(take_datetime_difference_in_years, args=(today_date, 'D')),
        Driver_experience_years=lambda df: df['Date_driving_licence_dt'].apply(take_datetime_difference_in_years, args=(today_date, 'D')),
    )
)
```

**Note**: Consider using the data collection end date (2018-12-31) instead of `today_date` for consistency with the original data context.

### 5.4 Resulting Feature Set

After engineering, we have the following variables:
| Category | Variables |
|----------|-----------|
| **Identifiers** | ID |
| **Dates (raw)** | Date_birth, Date_driving_licence |
| **Dates (converted)** | Date_birth_dt, Date_driving_licence_dt |
| **Policy Info** | Distribution_channel, Premium, Cost_claims_year |
| **Vehicle Specs** | Type_risk, Year_matriculation, Power, Cylinder_capacity, Value_vehicle, N_doors, Type_fuel, Length, Weight |
| **Location** | Area |
| **Driver Info** | Second_driver |
| **Engineered** | power_to_weight, Car_age_years, Driver_age_years, Driver_experience_years |
| **Targets** | Premium, Cost_claims_year, claims_frequency |
---

## 6. Feature Exploration

### 6.1 Feature Completeness Check

**Purpose**: Identify missing values before modeling.

```python
features_trainset.isnull().sum()
```

**Why This Matters**: Missing values can break models or bias predictions. Identifying them early allows for proper imputation strategies.

### 6.2 Continuous Variable Distributions

#### Manual Binning Pattern

**Purpose**: Understand the distribution of continuous variables.

```python
# Step 1: Bin the variable
variable = 'Weight'
binned_variable = pd.cut(features_trainset[variable], bins=10)

# Step 2: Join binned variable to dataset
binned_variable.name = f"binned_{variable}"
binned_df = pd.concat([features_trainset, binned_variable], axis=1)

# Step 3: Group by bins and count
groups = []
for group, subset in features_trainset.groupby(by=binned_variable, observed=False):
    groups.append({
        'Binrange': group,
        'Count': len(subset),
    })
group_df = pd.DataFrame(groups)

# Step 4: Visualize
plt.bar(x=group_df['Binrange'].astype(str), height=group_df['Count'])
plt.xticks(rotation=90)
plt.show()
```

**Why Manual Binning?**: Understanding the step-by-step process helps when you need custom bin definitions (e.g., age groups for pricing).

#### Scalable Histogram Pattern

**Purpose**: Quickly visualize multiple continuous variables.

```python
cols = ['Car_age_years', 'Driver_age_years', 'Driver_experience_years', 'Power', 'power_to_weight']
bin = 10

fig, axes = plt.subplots(1, len(cols), figsize=(18, 3))
for i, col in enumerate(cols):
    sns.histplot(data=features_trainset, x=col, bins=bin, kde=False, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()
```

**What to Look For**:

1. Skewness (may need log transformation)
2. Outliers (may need capping or investigation)
3. Multi-modal distributions (may indicate subgroups)

### 6.3 Categorical Variable Distributions

```python
cols = [
    'Distribution_channel',
    'Type_risk',
    'Type_fuel',
    'Area',
    'Second_driver'
]

fig, axes = plt.subplots(2, 3, figsize=(15, 6))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.countplot(data=features_trainset, x=col, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].tick_params(axis="x")
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()
```

**What to Look For**:

1. Class imbalance (may need stratified sampling)
2. Rare categories (may need grouping)
3. Unexpected distributions (may indicate data quality issues)

### 6.4 Bivariate Correlation Analysis

```python
features_trainset[['Power', 'Cylinder_capacity', 'power_to_weight', 'Value_vehicle', 'Length', 'Weight']].corr()
```

**Why This Matters**: Highly correlated features (multicollinearity) can:

1. Inflate variance in regression coefficients
2. Make feature importance unreliable
3. Cause overfitting

**Common Findings**:

1. Power and Cylinder_capacity are often highly correlated
2. Length and Weight are typically correlated (bigger cars are heavier)
3. Consider dropping one of highly correlated pairs

---

## 7. Conclusion

### Summary of Engineered Features
| Feature | Type | Business Meaning |
|---------|------|------------------|
| `Driver_age_years` | Continuous | Risk proxy: younger drivers have more accidents |
| `Driver_experience_years` | Continuous | Risk proxy: less experience means more accidents |
| `Car_age_years` | Continuous | Safety proxy: older cars lack modern safety features |
| `power_to_weight` | Continuous | Performance proxy: sportier cars mean riskier driving |

### Key Insights

1. **Domain Knowledge is Essential**: The best features come from understanding the business, not just the data
2. **Simple Features Often Work Best**: Age, experience, and car characteristics are tried-and-true predictors
3. **Missing Data Matters**: Some valuable features (occupation, medical history) aren't available at quote time
4. **Correlation Awareness**: Watch for multicollinearity among vehicle specifications

### What We Learned
1. **Feature engineering is iterative**: Start with domain knowledge, validate with data exploration
2. **Documentation matters**: Understanding *why* features exist helps future maintenance
3. **Keep it simple**: Four engineered features provide significant predictive lift
---

## Appendix: Technical Reference

### File Locations

```
Raw Data:
  - data/output/Insurance_Initiation_Variables.csv
  - data/input/exp/sample_type_claim.csv

Feature Engineering Notebook:
  - notebook/freq-sev-approach/feature.ipynb

Related Notebooks:
  - notebook/freq-sev-approach/dataset.ipynb
  - notebook/freq-sev-approach/explore.ipynb
```

### Dependencies

```python
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### Key Functions Used

| Function | Purpose | Example |
|----------|---------|---------|
| `pd.read_csv()` | Load CSV data | `pd.read_csv(path, delimiter=';')` |
| `df.merge()` | Join dataframes | `df1.merge(df2, on='col', how='left')` |
| `df.assign()` | Add new columns | `df.assign(new_col=df['col'] / 100)` |
| `df.apply()` | Apply function to column | `df['col'].apply(func)` |
| `pd.to_datetime()` | Convert to datetime | `pd.to_datetime(value, format='%d/%m/%Y')` |
| `pd.cut()` | Bin continuous variable | `pd.cut(df['col'], bins=10)` |
| `sns.histplot()` | Histogram visualization | `sns.histplot(data=df, x='col')` |
| `sns.countplot()` | Bar chart for categories | `sns.countplot(data=df, x='col')` |
| `df.corr()` | Correlation matrix | `df[['col1', 'col2']].corr()` |

### Recommended Reading

1. **Insurance Fundamentals**: [insurance-a-mind-dump-so-you-get-a-head-start](https://medium.com/@olumideodetunde/insurance-a-mind-dump-so-you-get-a-head-start-e1b8dbc4ce76)
---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Source Notebook**: `notebook/freq-sev-approach/feature.ipynb`  
**For Questions**: Refer to the feature engineering notebook for complete implementation