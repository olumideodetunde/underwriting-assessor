import pytest
import pandas as pd
from src.feature.driver import Driver


# --- Fixtures ---

@pytest.fixture
def driver():
    return Driver()


@pytest.fixture
def sample_df():
    """DataFrame mimicking the raw input with string date columns."""
    return pd.DataFrame({
        "Date_last_renewal": ["01/06/2020", "15/03/2021", "28/12/2019"],
        "Date_birth": ["01/06/1990", "15/03/1985", "28/12/1975"],
        "claim_amount": [100.0, 250.0, 50.0],
    })


@pytest.fixture
def datetime_df():
    """DataFrame with already-converted datetime columns."""
    return pd.DataFrame({
        "Date_last_renewal": pd.to_datetime(
            ["01/06/2020", "15/03/2021", "28/12/2019"], format="%d/%m/%Y"
        ),
        "Date_birth": pd.to_datetime(
            ["01/06/1990", "15/03/1985", "28/12/1975"], format="%d/%m/%Y"
        ),
        "claim_amount": [100.0, 250.0, 50.0],
    })


# =============================================================
# 1. Instantiation
# =============================================================

class TestDriverInstantiation:

    def test_driver_is_instance_of_base(self, driver):
        from src.feature.base import BaseFeatureTransformer
        assert isinstance(driver, BaseFeatureTransformer)

    def test_driver_instantiates_successfully(self, driver):
        assert isinstance(driver, Driver)


# =============================================================
# 2. convert_all_date_columns_to_datetime
# =============================================================

class TestConvertAllDateColumnsToDatetime:

    def test_converts_date_columns_to_datetime(self, driver, sample_df):
        result = driver.convert_all_date_columns_to_datetime(sample_df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date_last_renewal"])
        assert pd.api.types.is_datetime64_any_dtype(result["Date_birth"])

    def test_non_date_columns_unchanged(self, driver, sample_df):
        result = driver.convert_all_date_columns_to_datetime(sample_df)
        pd.testing.assert_series_equal(result["claim_amount"], sample_df["claim_amount"])

    def test_does_not_mutate_original_dataframe(self, driver, sample_df):
        original_dtype = sample_df["Date_last_renewal"].dtype
        _ = driver.convert_all_date_columns_to_datetime(sample_df)
        assert sample_df["Date_last_renewal"].dtype == original_dtype

    def test_returns_dataframe(self, driver, sample_df):
        result = driver.convert_all_date_columns_to_datetime(sample_df)
        assert isinstance(result, pd.DataFrame)


# =============================================================
# 3. create_driver_age_at_contract_inception
# =============================================================

class TestCreateDriverAgeAtContractInception:

    def test_creates_age_column(self, driver, datetime_df):
        result = driver.create_driver_age_at_contract_inception(datetime_df)
        assert "driver_age_at_contract_inception" in result.columns

    def test_age_is_positive_timedelta(self, driver, datetime_df):
        result = driver.create_driver_age_at_contract_inception(datetime_df)
        assert (result["driver_age_at_contract_inception"] >= pd.Timedelta(0)).all()

    def test_age_values_are_correct(self, driver, datetime_df):
        result = driver.create_driver_age_at_contract_inception(datetime_df)
        expected = abs(datetime_df["Date_last_renewal"] - datetime_df["Date_birth"])
        pd.testing.assert_series_equal(
            result["driver_age_at_contract_inception"], expected, check_names=False
        )

    def test_custom_column_name(self, driver, datetime_df):
        result = driver.create_driver_age_at_contract_inception(
            datetime_df, column_to_be_create="custom_age"
        )
        assert "custom_age" in result.columns
        assert "driver_age_at_contract_inception" not in result.columns

    def test_does_not_mutate_original_dataframe(self, driver, datetime_df):
        original_columns = list(datetime_df.columns)
        _ = driver.create_driver_age_at_contract_inception(datetime_df)
        assert list(datetime_df.columns) == original_columns

    def test_returns_dataframe(self, driver, datetime_df):
        result = driver.create_driver_age_at_contract_inception(datetime_df)
        assert isinstance(result, pd.DataFrame)


# =============================================================
# 4. transform (main entry point)
# =============================================================

class TestTransform:

    def test_transform_converts_dates_and_creates_age(self, driver, sample_df):
        result = driver.transform(sample_df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date_last_renewal"])
        assert pd.api.types.is_datetime64_any_dtype(result["Date_birth"])
        assert "driver_age_at_contract_inception" in result.columns

    def test_transform_returns_dataframe(self, driver, sample_df):
        result = driver.transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_does_not_mutate_original(self, driver, sample_df):
        original_columns = list(sample_df.columns)
        _ = driver.transform(sample_df)
        assert list(sample_df.columns) == original_columns

    def test_transform_preserves_non_date_columns(self, driver, sample_df):
        result = driver.transform(sample_df)
        pd.testing.assert_series_equal(result["claim_amount"], sample_df["claim_amount"])

    def test_transform_age_is_positive(self, driver, sample_df):
        result = driver.transform(sample_df)
        assert (result["driver_age_at_contract_inception"] >= pd.Timedelta(0)).all()

