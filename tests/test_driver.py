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
        "Date_driving_licence": ["01/06/2008", "15/03/2005", "28/12/1995"],
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
        "Date_driving_licence": pd.to_datetime(
            ["01/06/2008", "15/03/2005", "28/12/1995"], format="%d/%m/%Y"
        ),
        "claim_amount": [100.0, 250.0, 50.0],
    })


@pytest.fixture
def age_experience_df():
    """DataFrame with pre-computed age and experience columns for gap/ratio tests."""
    return pd.DataFrame({
        "driver_age_at_contract_inception": [30, 36, 44],
        "driver_experience_age": [12, 16, 24],
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

    def test_age_is_positive(self, driver, datetime_df):
        result = driver.create_driver_age_at_contract_inception(datetime_df)
        assert (result["driver_age_at_contract_inception"] >= 0).all()

    def test_age_values_are_correct(self, driver, datetime_df):
        result = driver.create_driver_age_at_contract_inception(datetime_df)
        expected = datetime_df["Date_last_renewal"].dt.year - datetime_df["Date_birth"].dt.year
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
# 4. create_driving_experience_length_in_years
# =============================================================

class TestCreateDrivingExperienceLengthInYears:

    def test_creates_experience_column(self, driver, datetime_df):
        result = driver.create_driving_experience_length_in_years(datetime_df)
        assert "driver_experience_age" in result.columns

    def test_experience_is_positive(self, driver, datetime_df):
        result = driver.create_driving_experience_length_in_years(datetime_df)
        assert (result["driver_experience_age"] >= 0).all()

    def test_experience_values_are_correct(self, driver, datetime_df):
        result = driver.create_driving_experience_length_in_years(datetime_df)
        expected = datetime_df["Date_last_renewal"].dt.year - datetime_df["Date_driving_licence"].dt.year
        pd.testing.assert_series_equal(
            result["driver_experience_age"], expected, check_names=False
        )

    def test_custom_column_name(self, driver, datetime_df):
        result = driver.create_driving_experience_length_in_years(
            datetime_df, column_to_be_created="custom_experience"
        )
        assert "custom_experience" in result.columns
        assert "driver_experience_age" not in result.columns

    def test_does_not_mutate_original_dataframe(self, driver, datetime_df):
        original_columns = list(datetime_df.columns)
        _ = driver.create_driving_experience_length_in_years(datetime_df)
        assert list(datetime_df.columns) == original_columns

    def test_returns_dataframe(self, driver, datetime_df):
        result = driver.create_driving_experience_length_in_years(datetime_df)
        assert isinstance(result, pd.DataFrame)


# =============================================================
# 5. create_driver_age_experience_gap
# =============================================================

class TestCreateDriverAgeExperienceGap:

    def test_creates_gap_column(self, driver, age_experience_df):
        result = driver.create_driver_age_experience_gap(
            age_experience_df,
            age_col="driver_age_at_contract_inception",
            experience_col="driver_experience_age",
        )
        assert "driver_age_experience_age_diff" in result.columns

    def test_gap_values_are_correct(self, driver, age_experience_df):
        result = driver.create_driver_age_experience_gap(
            age_experience_df,
            age_col="driver_age_at_contract_inception",
            experience_col="driver_experience_age",
        )
        expected = abs(age_experience_df["driver_age_at_contract_inception"]
                       - age_experience_df["driver_experience_age"])
        pd.testing.assert_series_equal(
            result["driver_age_experience_age_diff"], expected, check_names=False
        )

    def test_gap_is_always_positive(self, driver):
        df = pd.DataFrame({
            "driver_age_at_contract_inception": [10, 20],
            "driver_experience_age": [20, 10],
        })
        result = driver.create_driver_age_experience_gap(
            df,
            age_col="driver_age_at_contract_inception",
            experience_col="driver_experience_age",
        )
        assert (result["driver_age_experience_age_diff"] >= 0).all()

    def test_custom_column_name(self, driver, age_experience_df):
        result = driver.create_driver_age_experience_gap(
            age_experience_df,
            age_col="driver_age_at_contract_inception",
            experience_col="driver_experience_age",
            column_to_be_created="custom_gap",
        )
        assert "custom_gap" in result.columns
        assert "driver_age_experience_age_diff" not in result.columns

    def test_does_not_mutate_original_dataframe(self, driver, age_experience_df):
        original_columns = list(age_experience_df.columns)
        _ = driver.create_driver_age_experience_gap(
            age_experience_df,
            age_col="driver_age_at_contract_inception",
            experience_col="driver_experience_age",
        )
        assert list(age_experience_df.columns) == original_columns

    def test_returns_dataframe(self, driver, age_experience_df):
        result = driver.create_driver_age_experience_gap(
            age_experience_df,
            age_col="driver_age_at_contract_inception",
            experience_col="driver_experience_age",
        )
        assert isinstance(result, pd.DataFrame)


# =============================================================
# 6. create_driver_age_experience_ratio
# =============================================================

class TestCreateDriverAgeExperienceRatio:

    def test_creates_ratio_column(self, driver, age_experience_df):
        age_experience_df["driver_age_experience_age_diff"] = abs(
            age_experience_df["driver_age_at_contract_inception"]
            - age_experience_df["driver_experience_age"]
        )
        result = driver.create_driver_age_experience_ratio(
            age_experience_df,
            gap_col="driver_age_experience_age_diff",
            age_col="driver_age_at_contract_inception",
        )
        assert "driver_age_experience_ratio_proxy_for_driving_experience" in result.columns

    def test_ratio_values_are_correct(self, driver, age_experience_df):
        age_experience_df["driver_age_experience_age_diff"] = abs(
            age_experience_df["driver_age_at_contract_inception"]
            - age_experience_df["driver_experience_age"]
        )
        result = driver.create_driver_age_experience_ratio(
            age_experience_df,
            gap_col="driver_age_experience_age_diff",
            age_col="driver_age_at_contract_inception",
        )
        expected = (age_experience_df["driver_age_experience_age_diff"]
                    / age_experience_df["driver_age_at_contract_inception"])
        pd.testing.assert_series_equal(
            result["driver_age_experience_ratio_proxy_for_driving_experience"],
            expected, check_names=False,
        )

    def test_custom_column_name(self, driver, age_experience_df):
        age_experience_df["driver_age_experience_age_diff"] = [18, 20, 20]
        result = driver.create_driver_age_experience_ratio(
            age_experience_df,
            gap_col="driver_age_experience_age_diff",
            age_col="driver_age_at_contract_inception",
            column_to_be_created="custom_ratio",
        )
        assert "custom_ratio" in result.columns
        assert "driver_age_experience_ratio_proxy_for_driving_experience" not in result.columns

    def test_does_not_mutate_original_dataframe(self, driver, age_experience_df):
        age_experience_df["driver_age_experience_age_diff"] = [18, 20, 20]
        original_columns = list(age_experience_df.columns)
        _ = driver.create_driver_age_experience_ratio(
            age_experience_df,
            gap_col="driver_age_experience_age_diff",
            age_col="driver_age_at_contract_inception",
        )
        assert list(age_experience_df.columns) == original_columns

    def test_returns_dataframe(self, driver, age_experience_df):
        age_experience_df["driver_age_experience_age_diff"] = [18, 20, 20]
        result = driver.create_driver_age_experience_ratio(
            age_experience_df,
            gap_col="driver_age_experience_age_diff",
            age_col="driver_age_at_contract_inception",
        )
        assert isinstance(result, pd.DataFrame)

    def test_division_by_zero_produces_inf(self, driver):
        """If driver age is 0, ratio should be inf — not crash."""
        df = pd.DataFrame({
            "driver_age_experience_age_diff": [10],
            "driver_age_at_contract_inception": [0],
        })
        result = driver.create_driver_age_experience_ratio(
            df,
            gap_col="driver_age_experience_age_diff",
            age_col="driver_age_at_contract_inception",
        )
        assert result["driver_age_experience_ratio_proxy_for_driving_experience"].iloc[0] == float("inf")


# =============================================================
# 7. transform (main entry point)
# =============================================================

class TestTransform:

    def test_transform_converts_dates_and_creates_age(self, driver, sample_df):
        result = driver.transform(sample_df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date_last_renewal"])
        assert pd.api.types.is_datetime64_any_dtype(result["Date_birth"])
        assert "driver_age_at_contract_inception" in result.columns

    def test_transform_creates_all_feature_columns(self, driver, sample_df):
        result = driver.transform(sample_df)
        assert "driver_age_at_contract_inception" in result.columns
        assert "driver_experience_age" in result.columns
        assert "driver_age_experience_age_diff" in result.columns
        assert "driver_age_experience_ratio_proxy_for_driving_experience" in result.columns

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
        assert (result["driver_age_at_contract_inception"] >= 0).all()

    def test_transform_experience_is_positive(self, driver, sample_df):
        result = driver.transform(sample_df)
        assert (result["driver_experience_age"] >= 0).all()

