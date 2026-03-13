import pandas as pd

from src.feature.base import BaseFeatureTransformer

class Driver(BaseFeatureTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.convert_all_date_columns_to_datetime(df)
        df = self.create_driver_age_at_contract_inception(df)
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
        df = self._take_absolute_difference_between_datetime_columns(
            df, date_col, date_col_2, created_column=column_to_be_create
        )
        return df





