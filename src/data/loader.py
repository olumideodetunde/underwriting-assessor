import pandas as pd


def load_csv(path: str, delimiter: str = ";",
             na_values: list[str] | None = ["NA", ""],reader=pd.read_csv) -> pd.DataFrame:
    df = reader(path,delimiter=delimiter,na_values=na_values)
    return df

