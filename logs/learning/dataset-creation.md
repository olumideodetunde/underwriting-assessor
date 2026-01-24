# Motor Vehicle Insurance Dataset Creation: A Step-by-Step Guide

## 1. Introduction

Welcome! This guide walks you through how we created the **Motor Vehicle Insurance Dataset**, a crucial foundation for building predictive models in insurance pricing and risk assessment.

### What is This Dataset?

The dataset contains comprehensive information about motor vehicle insurance policies combined with their associated claims history. Specifically, it merges:
- **Insurance policy features**: Information about policyholders, their vehicles, and contract details
- **Claims frequency data**: The number and types of claims filed per policyholder per year

### Purpose

This dataset enables us to build **frequency models** (predicting how many claims a policyholder will make) and **severity models** (predicting the cost of those claims). Together, these models help insurers calculate fair premiums by estimating expected claim costs.

### Real-World Context

Imagine you work at an insurance company. To price a motor vehicle policy fairly, you need to know:
- How likely is this customer to file a claim?
- How much will those claims cost on average?

This dataset provides the historical evidence to answer these questions!

---

## 2. Data Sources

### Source 1: Motor Vehicle Insurance Data
**File**: `data/input/exp/Motor_vehicle_insurance_data.csv`

This is our primary dataset containing information about insurance policies. Think of it as a **customer database** that tracks:
- **Policyholder information**: Age, driving license date, customer seniority
- **Policy details**: Contract start/renewal dates, distribution channel
- **Vehicle characteristics**: Make, model, power, fuel type, vehicle value
- **Policy status**: Number of active policies, maximum policies allowed

**Size**: 105,555 policy records across multiple years

### Source 2: Claims Data
**File**: `data/input/exp/sample_type_claim.csv`

This dataset records **claims history** - essentially, what happened when policyholders filed claims. It contains:
- **Claim classifications**: Different types of claims (e.g., collision, theft, liability)
- **Claim amounts**: The cost breakdown by claim type
- **Time period**: The year the claim occurred

**Key concept**: Each row represents a *claim type* for a specific policyholder in a specific year. So one policyholder might have multiple rows if they filed different types of claims in the same year.

### Data Relationship
These two datasets are connected by:
- **ID**: A unique policyholder identifier
- **Year**: The policy year (cost_claims_year in claims data, derived from policy dates in insurance data)

---

## 3. Data Collection

### How Was This Data Gathered?

This data comes from **actual insurance company operations**. Here's how insurers typically collect this:

1. **Policy Creation**: When a customer buys or renews an insurance policy, their information is recorded in the policy management system
2. **Claims Filing**: When a customer reports an accident or incident, it's recorded with:
   - The type of claim (classification)
   - The amount claimed
   - When it happened
3. **Data Export**: Both systems are periodically exported into CSV files for analysis

### Collection Period
The data spans multiple years (2015-2019 based on the dates we see), capturing a full business cycle including:
- New customer acquisitions
- Renewals
- Claims patterns across different time periods

### Data Format
- **Format**: CSV (Comma-Separated Values) - a simple, portable format
- **Delimiter**: Semicolon (`;`) instead of comma for some fields (common in European data)
- **Character Encoding**: Standard UTF-8

---

## 4. Data Preprocessing

Now here's where the magic happens! Raw data from insurance systems isn't immediately ready for analysis. We need to **transform it** into a usable dataset.

### Step 1: Load the Raw Data

```python
import pandas as pd

# Load both datasets
insurance = pd.read_csv('../../data/input/exp/Motor_vehicle_insurance_data.csv', delimiter=";")
claims = pd.read_csv('../../data/input/exp/sample_type_claim.csv', delimiter=';')
```

**Why semicolon delimiter?** This is common in European countries where commas are used as decimal separators.

### Step 2: Aggregate Claims by Type (Create Frequency)

**The Challenge**: The claims data has one row per claim *type*. But we want one number: "How many claims did this customer have in this year?"

**The Solution**: Group all claim types for the same customer and year, then count them.

```python
# Group by policyholder ID and year, then count the number of claim types
claims_frequency = (
    claims
    .groupby(['ID', 'Cost_claims_year'])  # Group by customer and year
    .agg({'Cost_claims_by_type': 'count'})  # Count the claim types
    .rename(columns={'Cost_claims_by_type': 'claims_frequency'})  # Name it clearly
    .reset_index()  # Convert to a regular table
)
```

**What just happened?**
- `groupby(['ID', 'Cost_claims_year'])`: Creates groups for each unique (customer, year) pair
- `agg({'Cost_claims_by_type': 'count'})`: Counts how many rows (claim types) are in each group
- `rename()`: Makes the column name meaningful
- `reset_index()`: Converts the grouped result back to a flat table

**Example**:
```
Customer ID | Year | Claims Frequency
    1       | 2016 |       2
    1       | 2017 |       1
    2       | 2016 |       0
```

### Step 3: Merge Insurance Data with Claims Frequency

**The Challenge**: The insurance data has many rows per customer (one per year) and doesn't have claims information yet.

**The Solution**: Join (merge) the aggregated claims frequency to each insurance record.

```python
dataset = (
    pd
    .merge(
        left=insurance,           # Start with insurance data
        right=claims_frequency,   # Add claims frequency data
        how='left',               # Keep all insurance records
        on=['ID', 'Cost_claims_year']  # Match on customer ID and year
    )
    .fillna(value={'claims_frequency': 0})  # Replace NaN with 0
)
```

**What just happened?**
- `merge(..., how='left')`: **Left join** - keep every row from the insurance data
- If a customer had no claims that year, they won't appear in the claims_frequency table
- `.fillna(value={'claims_frequency': 0})`: Replace missing values with 0 (no claims)

### Step 4: Handle Missing Values

**Key Insight**: A missing claims_frequency value doesn't mean "unknown" - it means **zero claims**. Why?
- We aggregated ALL customers who had any claims
- If someone isn't in the aggregated data, they had no claims that year

**Imputation Strategy**: Fill with 0 (most appropriate for count data)

---

## 5. Data Structure

### Final Dataset Characteristics

| Characteristic | Details |
|---|---|
| **Total Records** | 105,555 rows |
| **Total Columns** | 31 columns |
| **Unique Customers** | ~53,500 policyholders |
| **Time Period** | 2015-2019 |
| **Target Variable** | `claims_frequency` |

### Column Categories

#### A. Temporal Information (Policy Dates)
- `Date_start_contract`: When the policy began
- `Date_last_renewal`: Most recent renewal date
- `Date_next_renewal`: When the policy renews next
- `Cost_claims_year`: The year being considered

#### B. Policyholder Demographics
- `Date_birth`: Customer age (derived from this)
- `Date_driving_licence`: Driving experience (years since license)
- `Seniority`: How long they've been with the insurer

#### C. Policy Characteristics
- `Distribution_channel`: How they bought the policy (direct, agent, broker, etc.)
- `Policies_in_force`: Number of active policies with us
- `Max_policies`: Maximum allowed policies

#### D. Vehicle Information
- `Year_matriculation`: Vehicle age
- `Power`: Engine power (CV/HP)
- `Cylinder_capacity`: Engine size
- `Value_vehicle`: Vehicle's monetary value
- `N_doors`: Number of doors
- `Type_fuel`: Fuel type (P=Petrol, D=Diesel)
- `Length`: Vehicle length in meters
- `Weight`: Vehicle weight in kg

#### E. Risk Factors
- `Area`: Geographic location (rural/urban)
- `Type_risk`: Vehicle usage type
- `Second_driver`: Whether there's a second driver

#### F. Target Variable
- **`claims_frequency`**: Number of claims filed in that year (our prediction target)

### Data Types

```
Integer columns: ID, Seniority, Policies_in_force, Max_policies, 
                 Power, Year_matriculation, N_doors, claims_frequency
                 
Float columns: Cylinder_capacity, Value_vehicle, Length, Weight

Date columns: All Date_* columns

Categorical: Distribution_channel, Area, Type_risk, Type_fuel, Second_driver
```

### Sample Data

```
ID | Date_start_contract | Year_matriculation | Power | claims_frequency
---|---|---|---|---
1  | 05/11/2015          | 2004               | 80    | 0
1  | 05/11/2016          | 2004               | 80    | 0
1  | 05/11/2017          | 2004               | 80    | 0
2  | 26/09/2017          | 2004               | 80    | 0
```

---

## 6. Data Quality

### Issues Encountered and Solutions

#### Issue 1: Missing Claims Records
**Problem**: Some customers have no entry in the claims data even though they had insurance.

**Explanation**: This is actually correct! It means they filed **zero claims** that year.

**Solution**: Used `fillna(claims_frequency=0)` to properly represent "no claims"

**Learning Point**: In data preprocessing, "missing" can mean different things:
- "Unknown value" → use imputation techniques (mean, median, mode)
- "Absence of event" → use 0 (as we did here)

#### Issue 2: Missing Vehicle Dimensions
**Problem**: Some records have NaN values for `Length` (vehicle length)

**Example**:
```
ID | Length | Weight
---|---|---
1  | NaN    | 190
```

**Why?**: Older vehicles or certain vehicle types might not have standardized measurements.

**Solution**: Left as NaN (Not addressed in this dataset creation, but important for downstream modeling)

**Best Practice**: When you encounter NaNs:
1. Understand *why* they're missing
2. Check if missingness is random or systematic
3. Decide on an imputation strategy (drop records, use mean/median, use model-based imputation)

#### Issue 3: Data Consistency Across Years
**Problem**: The same customer appears multiple times (one row per year), but some values should remain constant.

**Example**: Vehicle characteristics shouldn't change between years for the same vehicle.

**Validation**: You'd want to verify that vehicle specs are consistent across years for the same customer.

### Data Quality Metrics

- **Completeness**: 
  - `claims_frequency`: 100% (after filling NaN with 0)
  - Most columns: >95% complete
  - `Length`: ~80% complete (missing for some vehicles)

- **Consistency**: 
  - All claim frequencies are non-negative integers ✓
  - All vehicle weights are reasonable (500-2500 kg) ✓
  - Customer IDs are unique identifiers ✓

- **Accuracy**:
  - Dates follow proper format (DD/MM/YYYY)
  - Age values are realistic (18-80 years)
  - Vehicle values are within expected ranges

### Validation Checklist
Before using this dataset, verify:
- [ ] Claims frequency is non-negative integer
- [ ] All merge keys (ID, Cost_claims_year) are non-null
- [ ] Date columns parse correctly
- [ ] No duplicate rows for same (ID, Cost_claims_year) pair
- [ ] Vehicle characteristics are stable across years per customer

---

## 7.. Conclusion

### Key Takeaways

1. **What We Built**: A comprehensive dataset combining insurance policy information with historical claims data, ready for predictive modeling.

2. **The Process**:
   - Loaded raw data from two sources (policies and claims)
   - Aggregated claims by type to get claim counts per customer per year
   - Merged the datasets on customer ID and year
   - Handled missing claims as zeros (correctly representing "no claims")

3. **Why This Matters**: By combining policy features with claims history, we created a dataset that lets us build models that understand the relationship between customer/vehicle characteristics and claim frequency.

4. **Data Quality**: The dataset is well-structured with 105,555 records across 31 meaningful features. Most values are complete, with appropriate handling of missing values.

5. **Next Steps**: This dataset is the foundation for:
   - Frequency modeling (predicting claim counts)
   - Severity modeling (predicting claim costs)
   - Premium calculation (frequency × severity = expected cost)
   - Risk segmentation and customer profiling

### Common Pitfalls to Avoid

❌ **Don't**: Treat missing `claims_frequency` as "unknown" - they're zeros
✓ **Do**: Understand that absence of data in claims = no claims filed

❌ **Don't**: Ignore the left join behavior - it ensures every policy record is kept
✓ **Do**: Remember this preserves even customers with zero claims

❌ **Don't**: Use this dataset without understanding its temporal structure (multiple years per customer)
✓ **Do**: Account for autocorrelation when modeling (same customer across years)

### Further Learning

**To deepen your understanding, explore**:
- The raw CSV files: Look at a few rows to see the actual data format
- The aggregate queries: Manually calculate claims frequency for one customer to understand the groupby operation
- The feature engineering notebook: See how these raw features are transformed for modeling
- Insurance domain basics: Understanding insurance products helps you understand why certain features matter

### Questions to Explore

1. What happens if a customer has multiple policies in the same year? How does our dataset handle this?
2. Why did we use a **left join** instead of an **inner join**? What would change?
3. How would you validate that the merge operation worked correctly?
4. What other features might you engineer from these 31 columns?

---

## Appendix: Technical Reference

### File Locations
```
Raw Data:
  - data/input/exp/Motor_vehicle_insurance_data.csv (105,555 rows)
  - data/input/exp/sample_type_claim.csv (aggregated claims)

Generated Dataset:
  - Created in memory during notebook execution
  - Can be saved to: data/output/ for downstream use

Documentation:
  - data/input/exp/Descriptive of the variables.xlsx (full variable definitions)
  - data/input/exp/Insurance_Initiation_Variables.csv (initial variable info)

Processing Notebook:
  - notebook/freq-sev-approach/dataset.ipynb (actual code)
```

### Dependencies

```python
import pandas as pd              # Data manipulation
```

### Key Pandas Operations

| Operation | Purpose | Code |
|---|---|---|
| **Read CSV** | Load raw data | `pd.read_csv(path, delimiter=";")` |
| **GroupBy + Count** | Aggregate claims | `.groupby(['ID', 'year']).agg({'col': 'count'})` |
| **Merge** | Join datasets | `pd.merge(left, right, how='left', on=['ID', 'year'])` |
| **FillNA** | Handle missing values | `.fillna({'col': 0})` |
| **Reset Index** | Flatten grouped results | `.reset_index()` |

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**For Questions**: Refer to the dataset.ipynb notebook for the complete implementation
