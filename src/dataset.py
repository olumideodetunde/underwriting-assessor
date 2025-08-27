import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

class Dataset:

    def __init__(self, data_path:str|Path, claims_path:str|Path) -> None:
        self.claims_path = claims_path
        self.data_path  = data_path
        self.dataset = None
        self.train = None
        self.test  = None

    @staticmethod
    def load_data(path:str | Path ) -> pd.DataFrame:
        if str(path).endswith('.csv'):
            return pd.read_csv(path, delimiter=';')
        else:
            raise ValueError('Not a CSV file')

    ##TODO: create dataset bloating up the merged dataset
    def create_dataset(self, merge_columns:list) -> pd.DataFrame:
        data = self.load_data(self.data_path)
        claims = self.load_data(self.claims_path)
        self.dataset = pd.merge(
            left=data,
            right=claims,
            how = 'left',
            on = merge_columns
        )
        return self.dataset

    def split_dataset(self):
        pass


if __name__ == '__main__':
    rating_csv = "/Users/olumide/Library/CloudStorage/OneDrive-Personal/Documents/Research/Project 1/underwriting assessor/data/input/exp/Motor_vehicle_insurance_data.csv"
    clams_csv = "/Users/olumide/Library/CloudStorage/OneDrive-Personal/Documents/Research/Project 1/underwriting assessor/data/input/exp/sample type claim.csv"
    DatasetInstance = Dataset(data_path=rating_csv, claims_path=clams_csv)
    merged = DatasetInstance.create_dataset(merge_columns=['ID','Cost_claims_year'])
    pd.set_option('display.max_columns', None)
    print(merged)
