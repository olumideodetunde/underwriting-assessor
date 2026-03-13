import numpy as np
import pandas as pd

from src.feature.base import BaseFeatureTransformer


class Vehicle(BaseFeatureTransformer):

    def __init__(self):
        ##kept here so that it can applied to test at some point using fit-transform?
        self.fuel_type_uniques: pd.Index | None = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.encode_fuel_type(df)
        df = self.log_transform_vehicle_value(df)
        return df

    def encode_fuel_type(
            self,
            df: pd.DataFrame,
            source_col: str = "Type_fuel",
            column_to_be_created: str = "fuel_type_encoded") -> pd.DataFrame:
        df = df.copy()
        codes, uniques = pd.factorize(df[source_col])
        df[column_to_be_created] = codes
        self.fuel_type_uniques = uniques
        return df

    def log_transform_vehicle_value(
            self,
            df: pd.DataFrame,
            source_col: str = "Value_vehicle",
            column_to_be_created: str = "Value_vehicle_log_transformed") -> pd.DataFrame:
        df = df.copy()
        if (df[source_col] <= 0).any():
            raise ValueError(
                f"Column '{source_col}' contains zero or negative values. "
                "Log transform is undefined for these values."
            )
        df[column_to_be_created] = np.log(df[source_col])
        return df

