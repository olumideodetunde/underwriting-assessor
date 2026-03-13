import pytest
import pandas as pd
from src.feature.base import BaseFeatureTransformer


# --- Concrete child class for testing ---

class DummyTransformer(BaseFeatureTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


# --- Fixtures ---

@pytest.fixture
def transformer():
    return DummyTransformer()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "inception_date": ["01/06/2020", "15/03/2021", "28/12/2019"],
        "expiry_date": ["01/06/2021", "15/03/2022", "28/12/2020"],
        "claim_amount": [100.0, 250.0, 50.0],
    })


@pytest.fixture
def datetime_df():
    return pd.DataFrame({
        "inception_date": pd.to_datetime(
            ["01/06/2020", "15/03/2021", "28/12/2019"], format="%d/%m/%Y"
        ),
        "claim_amount": [100.0, 250.0, 50.0],
    })


# =============================================================
# 1. Abstract contract tests
# =============================================================

class TestAbstractContract:

    def test_cannot_instantiate_base_class_directly(self):
        with pytest.raises(TypeError):
            BaseFeatureTransformer()

    def test_child_missing_transform_raises_error(self):
        class IncompleteTransformer(BaseFeatureTransformer):
            pass

        with pytest.raises(TypeError):
            IncompleteTransformer()

    def test_concrete_child_instantiates_successfully(self, transformer):
        assert isinstance(transformer, BaseFeatureTransformer)


# =============================================================
# 2. _convert_column_to_datetime tests
# =============================================================

class TestConvertColumnToDatetime:

    def test_converts_single_column(self, transformer, sample_df):
        result = transformer._convert_column_to_datetime(
            sample_df, date_column=["inception_date"]
        )
        assert pd.api.types.is_datetime64_any_dtype(result["inception_date"])

    def test_converts_multiple_columns(self, transformer, sample_df):
        result = transformer._convert_column_to_datetime(
            sample_df, date_column=["inception_date", "expiry_date"]
        )
        assert pd.api.types.is_datetime64_any_dtype(result["inception_date"])
        assert pd.api.types.is_datetime64_any_dtype(result["expiry_date"])

    def test_parses_day_first_correctly(self, transformer, sample_df):
        result = transformer._convert_column_to_datetime(
            sample_df, date_column=["inception_date"], day_first=True
        )
        assert result["inception_date"].iloc[0].month == 6
        assert result["inception_date"].iloc[0].day == 1

    def test_invalid_dates_become_nat(self, transformer):
        df = pd.DataFrame({"date_col": ["01/06/2020", "not_a_date", "28/12/2019"]})
        result = transformer._convert_column_to_datetime(df, date_column=["date_col"])
        assert pd.isna(result["date_col"].iloc[1])

    def test_does_not_mutate_original_dataframe(self, transformer, sample_df):
        original_dtype = sample_df["inception_date"].dtype
        _ = transformer._convert_column_to_datetime(
            sample_df, date_column=["inception_date"]
        )
        assert sample_df["inception_date"].dtype == original_dtype

    def test_returns_dataframe(self, transformer, sample_df):
        result = transformer._convert_column_to_datetime(
            sample_df, date_column=["inception_date"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_non_date_columns_unchanged(self, transformer, sample_df):
        result = transformer._convert_column_to_datetime(
            sample_df, date_column=["inception_date"]
        )
        pd.testing.assert_series_equal(result["claim_amount"], sample_df["claim_amount"])

    def test_custom_date_format(self, transformer):
        df = pd.DataFrame({"date_col": ["2020-06-01", "2021-03-15"]})
        result = transformer._convert_column_to_datetime(
            df, date_column=["date_col"], date_format="%Y-%m-%d", day_first=False
        )
        assert result["date_col"].iloc[0] == pd.Timestamp("2020-06-01")


# =============================================================
# 3. _extract_year_from_a_datetime_column tests
# =============================================================

class TestExtractYearFromDatetimeColumn:

    def test_extracts_year_correctly(self, transformer, datetime_df):
        result = transformer._extract_year_from_a_datetime_column(
            datetime_df, date_column="inception_date",
            extracted_year_column_name="inception_year"
        )
        assert list(result["inception_year"]) == [2020, 2021, 2019]

    def test_new_column_is_created(self, transformer, datetime_df):
        result = transformer._extract_year_from_a_datetime_column(
            datetime_df, date_column="inception_date",
            extracted_year_column_name="inception_year"
        )
        assert "inception_year" in result.columns

    def test_does_not_mutate_original_dataframe(self, transformer, datetime_df):
        original_columns = list(datetime_df.columns)
        _ = transformer._extract_year_from_a_datetime_column(
            datetime_df, date_column="inception_date",
            extracted_year_column_name="inception_year"
        )
        assert list(datetime_df.columns) == original_columns

    def test_returns_dataframe(self, transformer, datetime_df):
        result = transformer._extract_year_from_a_datetime_column(
            datetime_df, date_column="inception_date",
            extracted_year_column_name="inception_year"
        )
        assert isinstance(result, pd.DataFrame)

    def test_original_columns_preserved(self, transformer, datetime_df):
        result = transformer._extract_year_from_a_datetime_column(
            datetime_df, date_column="inception_date",
            extracted_year_column_name="inception_year"
        )
        for col in datetime_df.columns:
            assert col in result.columns

    def test_nat_values_produce_nan_year(self, transformer):
        df = pd.DataFrame({
            "date_col": pd.to_datetime(["01/06/2020", None], format="%d/%m/%Y")
        })
        result = transformer._extract_year_from_a_datetime_column(
            df, date_column="date_col", extracted_year_column_name="year"
        )
        assert result["year"].iloc[0] == 2020.0
        assert pd.isna(result["year"].iloc[1])

