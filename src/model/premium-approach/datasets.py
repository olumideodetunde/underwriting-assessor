## Dataset Module
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

## load datasets
insurance = pd.read_csv('../../data/input/exp/Motor_vehicle_insurance_data.csv', delimiter=";")
claims =  pd.read_csv('../../data/input/exp/sample_type_claim.csv', delimiter=';')


## columns to drop which won't be available for new customers
cols_to_drop = ['ID','Date_last_renewal','Date_start_contract','Date_next_renewal', 
                'Seniority', 'Policies_in_force', 'Max_policies', 
                'Max_products','Lapse', 'Date_lapse', 'Cost_claims_year', 
                'N_claims_year', 'N_claims_history', 'R_Claims_history']

insurance_new = insurance.drop(columns=cols_to_drop)

### Categorical Variables
# 1. Distribution Channel - 0: Agent, 1: Broker
# 2. Payment - 0: half-yearly, 1: annually
# 3. Type_risk - 1: motorbikes, 2:vans, 3:passenger cars, 4:agricultural vehicles
# 4. Area - 0:rural, 1:urban
# 5. Second_driver - 0: no, 1: yes
# 6. Type_fuel - P: petrol, D: diesel

# Function to map Distribution Channel
def distribution_channel(x):
    if x == 0:
        return 'Agent'
    else:
        return 'Broker'

# Function to map Payment
def payment_method(x):
    if x == 0:
        return 'half-yearly'
    else:
        return 'annually'
    
# Function to map Type_risk
def type_risk(x):
    if x == 1:
        return 'motorbikes'
    elif x == 2:
        return 'vans'
    elif x == 3:
        return 'passenger cars'
    else:
        return 'agricultural vehicles'
    
# Function to map Area
def area_type(x):
    if x == 0:
        return 'rural'
    else:
        return 'urban'
    
# Function to map Second_driver
def second_driver(x):
    if x == 0:
        return 'No'
    else:
        return 'Yes'
    
# Function to map Type_fuel
def type_fuel(x):
    if x == 'P':
        return 'petrol'
    else:
        return 'diesel'
    
# Apply the functions to the insurance dataframe
insurance_new['Distribution_channel'] = insurance_new['Distribution_channel'].apply(distribution_channel)
insurance_new['Payment'] = insurance_new['Payment'].apply(payment_method)
insurance_new['Type_risk'] = insurance_new['Type_risk'].apply(type_risk)    
insurance_new['Area'] = insurance_new['Area'].apply(area_type)
insurance_new['Second_driver'] = insurance_new['Second_driver'].apply(second_driver)
insurance_new['Type_fuel'] = insurance_new['Type_fuel'].apply(type_fuel)

## convert date columns to datetime format for variables that start with 'Date_'
date_columns = [col for col in insurance_new.columns if col.startswith('Date_')]
for col in date_columns:
    insurance_new[col] = pd.to_datetime(insurance_new[col],format='%d/%m/%Y', errors='coerce')
    
# Display the first few rows of the final dataset
if __name__ == "__main__":
    print("Dataset module loaded successfully.")
    print(insurance_new.head())