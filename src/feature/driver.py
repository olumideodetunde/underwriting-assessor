import pandas as pd

from src.feature.base import BaseFeatureTransformer

class Driver(BaseFeatureTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.convert_all_date_columns_to_datetime(df)
        df = self.create_driver_age_at_contract_inception(df)
        df = self.create_driving_experience_length_in_years(df)
        df = self.create_driver_age_experience_gap(df)
        df = self.create_driver_age_experience_ratio(df)
        return df

    def convert_all_date_columns_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols = df.columns[df.columns.str.startswith("Date")].tolist()
        df = self._convert_column_to_datetime(df, date_column=date_cols)
        return df


    def create_driver_age_at_contract_inception(self,
                                                df: pd.DataFrame,
                                                date_col: str='Date_last_renewal',
                                                date_col_2:str='Date_birth',
                                                column_to_be_create:str="driver_age_at_contract_inception") -> pd.DataFrame:
        df = self._take_absolute_difference_between_date_columns_in_years(
            df, date_col, date_col_2, created_column=column_to_be_create
        )
        return df

    def create_driving_experience_length_in_years(
            self,
            df: pd.DataFrame,
            date_col: str = 'Date_last_renewal',
            date_col_2: str = 'Date_driving_licence',
            column_to_be_created: str = "driver_experience_age") -> pd.DataFrame:
        df = self._take_absolute_difference_between_date_columns_in_years(
            df, date_col, date_col_2, created_column=column_to_be_created)
        return df

    def create_driver_age_experience_gap(
            self,
            df: pd.DataFrame,
            age_col: str = "driver_age_at_contract_inception",
            experience_col: str = "driver_experience_age",
            column_to_be_created: str = "driver_age_experience_age_diff") -> pd.DataFrame:
        df = df.copy()
        df[column_to_be_created] = abs(df[age_col] - df[experience_col])
        return df

    def create_driver_age_experience_ratio(
            self,
            df: pd.DataFrame,
            gap_col: str = "driver_age_experience_age_diff",
            age_col: str = "driver_age_at_contract_inception",
            column_to_be_created: str = "driver_age_experience_ratio_proxy_for_driving_experience") -> pd.DataFrame:
        df = df.copy()
        df[column_to_be_created] = df[gap_col] / df[age_col]
        return df






