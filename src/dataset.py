
import pandas as pd
from dotenv.variables import Literal
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, data_path:str, claims_path:str):
        self.claims = pd.read_csv(claims_path, delimiter=';')
        self.data  = pd.read_csv(data_path, delimiter=';')
        self.grouped_claims = None
        self.dataset = None
        self.train = None
        self.test  = None

    def group_claims(self, grouping_columns:list, aggregation_column:str, aggregation_method:str='count'):
        self.grouped_claims = (self.claims
                .groupby(grouping_columns)
                .agg({aggregation_column:aggregation_method})
                .rename(columns={aggregation_column:'claims_frequency'})
                .reset_index()
                )
        return self

    def create_dataset(self, merge_columns:list, join_method='left'):
        self.dataset = pd.merge(
            left=self.data,
            right=self.grouped_claims,
            on = merge_columns,
            how=join_method,
        )
        return self

    def split_dataset(self, test_ratio:float, to_shuffle:bool=False):
        self.train, self.test = train_test_split(
            self.dataset,
            test_size = test_ratio,
            shuffle = to_shuffle,
        )
        return self.train, self.test


if __name__ == '__main__':
    rating_csv = "../data/input/exp/Motor_vehicle_insurance_data.csv"
    claims_csv =  "../data/input/exp/sample_type_claim.csv"
    DatasetInstance = (Dataset(data_path=rating_csv, claims_path=claims_csv)
                  .group_claims(grouping_columns=['ID', 'Cost_claims_year'],
                               aggregation_column='Cost_claims_by_type')
                  .create_dataset(merge_columns=['ID', 'Cost_claims_year']))
    train, test = DatasetInstance.split_dataset(test_ratio=0.2, to_shuffle=True)
