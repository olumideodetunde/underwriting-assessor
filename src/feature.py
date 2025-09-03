import pandas as pd
import numpy as np
from datetime import datetime

class Feature:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    @staticmethod
    def take_int_difference(first_number: int, second_number: int) -> int:
        return abs(first_number - second_number)

    @staticmethod
    def take_datetime_difference_in_years(first_datetime: datetime, second_datetime: datetime, interval) -> float:
        diff = (second_datetime - first_datetime) / np.timedelta64(1, interval)
        diff_years = diff / 365.25
        return diff_years

    @staticmethod
    def convert_to_datetime(value: object, format: str = "%d/%m/%Y", yearfirst: bool = True) -> pd.Series:
        return pd.to_datetime(arg=value, format=format, yearfirst=yearfirst)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass