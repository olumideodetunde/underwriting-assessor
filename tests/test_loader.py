import pytest

from src.data.loader import load_csv

@pytest.fixture()
def sample_policy_csv(tmp_path) -> str:
    content = (
        "ID;Date_birth;Date_driving_licence;Cost_claims_year;Power;Length;Weight\n"
        "1;15/04/1956;20/03/1976;0;80;NA;190\n"
        "1;15/04/1956;20/03/1976;1;80;400;190\n"
        "2;01/01/1990;15/06/2010;0;120;420;210\n"
        "3;25/12/1985;10/11/2005;0;95;380;175\n"
    )
    csv_path = tmp_path / "policy.csv"
    csv_path.write_text(content)
    return str(csv_path)


class TestLoadCsv:

    def test_loads_correct_shape(self, sample_policy_csv):
        df = load_csv(sample_policy_csv)
        assert df.shape == (4, 7)

    def test_semicolon_delimiter(self, sample_policy_csv):
        df = load_csv(sample_policy_csv)
        assert "ID" in df.columns
        assert "Power" in df.columns

    def test_na_string_treated_as_nan(self, sample_policy_csv):
        df = load_csv(sample_policy_csv)
        assert df["Length"].isna().sum() == 1

    def test_numeric_columns_intact(self, sample_policy_csv):
        df = load_csv(sample_policy_csv)
        assert df["Power"].dtype in ["int64", "float64"]
        assert df["Weight"].dtype in ["int64", "float64"]