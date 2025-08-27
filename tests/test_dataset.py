import pytest
import pandas as pd
from src.dataset import Dataset




class TestDataset:

    def test_load_data_with_csv_file(self, tmp_path):
        path = tmp_path / 'dataset.csv'
        content = "col1;col2;col3;col4;col5;col6"
        path.write_text(content)
        df = Dataset.load_data(path)
        assert list(df.columns) == ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']

    def test_load_data_with_not_a_csv_file(self, tmp_path):
        path = tmp_path / 'dataset.xlsx'
        with pytest.raises(ValueError):
            Dataset.load_data(path)

    def test_create_dataset(self, tmp_path):
        # Create dummy data CSV
        data_csv = tmp_path / "data.csv"
        claims_csv = tmp_path / "claims.csv"

        data_df = pd.DataFrame({
            "ID": [1, 2],
            "Cost_claims_year": [2020, 2021],
            "Value": [100, 200]
        })
        claims_df = pd.DataFrame({
            "ID": [1, 2],
            "Cost_claims_year": [2020, 2021],
            "ClaimAmount": [500, 600]
        })

        data_df.to_csv(data_csv, sep=";", index=False)
        claims_df.to_csv(claims_csv, sep=";", index=False)

        ds = Dataset(data_path=data_csv, claims_path=claims_csv)
        merged = ds.create_dataset(merge_columns=["ID", "Cost_claims_year"])

        # Assertions
        assert "ClaimAmount" in merged.columns
        assert merged.loc[merged["ID"] == 1, "ClaimAmount"].iloc[0] == 500
        assert merged.loc[merged["ID"] == 2, "Value"].iloc[0] == 200
        assert len(merged) == 2

