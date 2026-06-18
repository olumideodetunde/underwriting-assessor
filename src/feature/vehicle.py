import numpy as np
import pandas as pd

from src.feature.base import BaseFeatureTransformer


class Vehicle(BaseFeatureTransformer):

    def __init__(self):
        self.fuel_type_uniques_: pd.Index | None = None

    def fit(self, df: pd.DataFrame, source_col: str = "Type_fuel") -> "Vehicle":
        # Learn the fuel-type -> code mapping ONCE, from training data only.
        _, uniques = pd.factorize(df[source_col])
        self.fuel_type_uniques_ = uniques
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.encode_fuel_type(df)
        df = self.log_transform_vehicle_value(df)
        return df

    def encode_fuel_type(
            self,
            df: pd.DataFrame,
            source_col: str = "Type_fuel",
            column_to_be_created: str = "fuel_type_encoded") -> pd.DataFrame:
        if self.fuel_type_uniques_ is None:
            raise RuntimeError(
                "Vehicle must be fitted before transform; call fit() first."
            )
        df = df.copy()
        # Apply the LEARNED mapping. Unseen categories -> -1.
        df[column_to_be_created] = self.fuel_type_uniques_.get_indexer(df[source_col])
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

