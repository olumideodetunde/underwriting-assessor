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
    def take_datetime_difference_in_years(first_datetime: datetime, second_datetime: datetime, interval:str) -> float:
        #Interval here must be 'D' although, other code exist as captured here - https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
        if interval is not 'D':
            raise ValueError('Only "D" interval is supported')
        diff = (second_datetime - first_datetime) / np.timedelta64(1, interval)
        diff_years = abs(diff / 365.25)
        return diff_years

    @staticmethod
    def convert_to_datetime(value: object, format: str = "%d/%m/%Y", yearfirst: bool = True) -> pd.Series:
        return pd.to_datetime(arg=value, format=format, yearfirst=yearfirst)

def main(df_to_be_engineered:pd.DataFrame) -> pd.DataFrame:
    today_date = pd.Timestamp.today()
    today_year = today_date.year
    df_with_engineered_features = (
        df_to_be_engineered
        .copy()
        .assign(
            Date_birth_dt=df_to_be_engineered['Date_birth'].apply(Feature.convert_to_datetime),
            Date_driving_licence_dt=df_to_be_engineered['Date_driving_licence'].apply(Feature.convert_to_datetime),
            power_to_weight=df_to_be_engineered['Power'] / df_to_be_engineered['Weight'],
            Car_age_years=df_to_be_engineered['Year_matriculation'].apply(Feature.take_int_difference, args=(today_year,))

        )
        .assign(
            Driver_age_years=lambda df: df['Date_birth_dt'].apply(Feature.take_datetime_difference_in_years,
                                                                  args=(today_date, 'D')),
            Driver_experience_years=lambda df: df['Date_driving_licence_dt'].apply(Feature.take_datetime_difference_in_years,
                                                                                   args=(today_date, 'D')),
        )
    )
    return df_with_engineered_features
