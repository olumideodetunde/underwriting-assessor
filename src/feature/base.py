import pandas as pd
from typing import List
from abc import ABC, abstractmethod

class BaseFeatureTransformer(ABC):

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def _convert_column_to_datetime(
                                    df:pd.DataFrame,
                                   date_column:List[str]|List[object],
                                   day_first:bool=True,
                                   date_format:str='%d/%m/%Y')->pd.DataFrame:
        df = df.copy()
        for col in date_column:
            df[col] = pd.to_datetime(df[col],
                                     format=date_format,
                                     dayfirst=day_first,
                                     errors='coerce')
        return df

    @staticmethod
    def _extract_year_from_a_datetime_column(df:pd.DataFrame,
                                    date_column:str,
                                    extracted_year_column_name:str)->pd.DataFrame:
        df = df.copy()
        df[extracted_year_column_name] = df[date_column].dt.year
        return df

    @staticmethod
    def _take_absolute_difference_between_datetime_columns(df:pd.DataFrame,
                                                           date_column_1:str,
                                                           date_time_column_2:str,
                                                           created_column:str)->pd.DataFrame:
        df = df.copy()
        if not (pd.api.types.is_datetime64_any_dtype(df[date_column_1])
                and pd.api.types.is_datetime64_any_dtype(df[date_time_column_2])):
            raise ValueError('These are not columns of datatype datetime')
        df[created_column] = abs(df[date_column_1] - df[date_time_column_2])
        return df



