import pandas as pd
import numpy as np
from datasets import  insurance_new

def feature_engineering(df):
    
    # Small constant to avoid division by zero
    epsilon = 1e-10  # Small constant to avoid division by zero
    
    # Create a copy of dataset to perform feature engineering
    df_cp = df.copy()   
       
    # Drop duplicated rows if any
    df_cp = df_cp.drop_duplicates().reset_index(drop=True)
    
    ## Time and Age Related Variables
    df_cp["PolicyHolder_Age"] = 2018 - df_cp["Date_birth"].dt.year # Create a new feature 'Age' from 'Date_birth'
    df_cp["Driving_experience_years"] = 2018 - df_cp["Date_driving_licence"].dt.year # Driving_experience_years
    df_cp["Age_at_license"] = df_cp["Date_driving_licence"].dt.year - df_cp["Date_birth"].dt.year # Age_at_license 
    df_cp["Young_driver_flag"] = (df_cp["PolicyHolder_Age"] < 25).astype(int).map({0: "No", 1: "Yes"}) # Young_driver_flag
    df_cp["Inexperienced_driver_flag"] = (df_cp["Driving_experience_years"] < 3).astype(int).map({0: "No", 1: "Yes"}) # Inexperienced_driver_flag
    
    #If second driver exists and primary driver is young. risk jumps.
    df_cp["Second_driver_risk_proxy"] = np.where(
        (df_cp["Second_driver"] == "Yes") & (df_cp["Young_driver_flag"] == "Yes"),
        "High Risk",
        "Normal Risk"
    )


    ## Financial Presures
    # High_value_vehicle_flag = Value_vehicle > percentile threshold
    df_cp["High_value_vehicle_flag"] = ((df_cp["Value_vehicle"] >
                                                df_cp["Value_vehicle"]
                                                .quantile(0.75))
                                            .astype(int).map({0: "No", 1: "Yes"}))

    ## Vehicle Related Variables
    
   ## Powertrain & performance
    df_cp["Vehicle_age"] = 2018 - df_cp["Year_matriculation"]  # Vehicle_age     
    df_cp["Power_to_weight_ratio"] = df_cp["Power"] / (df_cp["Weight"] + epsilon) # Power_to_weight_ratio     
    df_cp["Engine_intensity"] = df_cp["Power"] / (df_cp["Cylinder_capacity"] + epsilon) # Engine_intensity 
    # Large_vehicle_flag 
    df_cp["Large_vehicle_flag"] = ((df_cp["Weight"] > df_cp["Weight"].quantile(0.75)) |
                                        (df_cp["Length"] > df_cp["Length"].quantile(0.75))
                                        ).astype(int).map({0: "No", 1: "Yes"}) 
    # Sporty_vehicle_flag 
    df_cp["Sporty_vehicle_flag"] = (df_cp["Power_to_weight_ratio"] >
                                        df_cp["Power_to_weight_ratio"].quantile(0.75)
                                        ).astype(int).map({0: "No", 1: "Yes"}) 
    ## Vehicle age, value & depreciation
    df_cp["Log_vehicle_value"] = np.log1p(df_cp["Value_vehicle"]) # Log_vehicle_value      
    df_cp["Value_per_year"] = df_cp["Value_vehicle"] / (df_cp["Vehicle_age"] + 1) #  Value_per_year    
    df_cp["New_vehicle_flag"] = (df_cp["Vehicle_age"] <= 1).astype(int).map({0: "No", 1: "Yes"}) # New_vehicle_flag (≤1 year)    
    df_cp["Old_vehicle_flag"] = (df_cp["Vehicle_age"] > 10).astype(int).map({0: "No", 1: "Yes"}) # Old_vehicle_flag (>10 years)
    
    ## Risk Related Vehicle Proxies
    df_cp[" Engine_size_class"] = pd.cut(df_cp["Cylinder_capacity"],
                                         bins=[0, 1000, 1500, 2000, 3000, np.inf],
                                         labels=["Very Small", "Small", "Medium", "Large", "Very Large"]) # 
    df_cp["High_power_flag"] = (df_cp["Power"] >= df_cp["Power"].quantile(0.9)).astype(int).map({0: "No", 1: "Yes"}) # High_power_flag (top decile power)
    df_cp["Fuel_risk_bucket"] = df_cp["Type_fuel"].map({
        "diesel": "Higher Risk",
        "petrol": "Normal Risk"
    }) # Fuel_risk_bucket
    
    df_cp[" Vehicle_size_proxy"] = df_cp["Length"] * df_cp["Weight"]  # Vehicle_size_proxy = Length × Weight
    
    # Interaction features (where models actually make money)
    # Young_driver × High_power
    df_cp["Young_driver_High_power"] = np.where(
        (df_cp["Young_driver_flag"] == "Yes") & (df_cp["High_power_flag"] == "Yes"),
        "High Risk",
        "Normal Risk"
    )
    # Inexperienced_driver × Sporty_vehicle
    df_cp["Inexperienced_driver_Sporty_vehicle"] = np.where(
        (df_cp["Inexperienced_driver_flag"] == "Yes") & (df_cp["Sporty_vehicle_flag"] == "Yes"),
        "High Risk",
        "Normal Risk"
    )
    # Inexperienced_driver × Large_vehicle
    df_cp["Inexperienced_driver_Large_vehicle"] = np.where(
        (df_cp["Inexperienced_driver_flag"] == "Yes") & (df_cp["Large_vehicle_flag"] == "Yes"),
        "High Risk",
        "Normal Risk"
    )
    
    # Urban_area × High_value
    df_cp["Urban_area_High_value"] = np.where(
        (df_cp["Area"] == "urban") & (df_cp["High_value_vehicle_flag"] == "Yes"),
        "High Risk",
        "Normal Risk"
    ) 
    # half_yearly_payment × Young_driver  
    df_cp["half_yearly_payment_Young_driver"] = np.where(
        (df_cp["Payment"] == 'half-yearly') & (df_cp["Young_driver_flag"] == "Yes"),
        "High Risk",
        "Normal Risk"
    )
    # Second_driver × Young_driver
    df_cp["Second_driver_Young_driver"] = np.where(
        (df_cp["Second_driver"] == "Yes") & (df_cp["Young_driver_flag"] == "Yes"),
        "High Risk",
        "Normal Risk"
    )
  
    ## drop date columns
    date_columns = [col for col in df_cp.columns if col.startswith('Date_')]
    df_cp = df_cp.drop(columns=date_columns)
    
    # Convert 'Year_matriculation' to string type
    df_cp["Year_matriculation"] = df_cp["Year_matriculation"].astype("str")
        
    # Separate features and target variable
    feature = df_cp.drop(columns=['Premium'])
    # Apply log transformation to the target variable 'Premium'
    target = np.log1p(df_cp[['Premium']])
    
    # Return the feature set and target variable
    return feature, target


if __name__ == "__main__":
    feature, target = feature_engineering(insurance_new)
    print("Feature Engineering Completed.")
    print(f"Feature Set Shape: {feature.shape[0]:,} rows and {feature.shape[1]:,} columns")
    print(f"Target Variable Shape: {target.shape[0]:,} rows and {target.shape[1]:,} columns")   
      
