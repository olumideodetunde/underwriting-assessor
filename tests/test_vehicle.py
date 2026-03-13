import numpy as np
import pytest
import pandas as pd

from src.feature.vehicle import Vehicle


# --- Fixtures ---

@pytest.fixture
def vehicle():
    return Vehicle()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Type_fuel": ["Petrol", "Diesel", "Petrol", "LPG", "Diesel"],
        "Value_vehicle": [15000.0, 25000.0, 8000.0, 12000.0, 30000.0],
        "N_doors": [4, 4, 2, 4, 4],
    })


# =============================================================
# 1. Instantiation
# =============================================================

class TestVehicleInstantiation:

    def test_vehicle_is_instance_of_base(self, vehicle):
        from src.feature.base import BaseFeatureTransformer
        assert isinstance(vehicle, BaseFeatureTransformer)

    def test_vehicle_instantiates_successfully(self, vehicle):
        assert isinstance(vehicle, Vehicle)

    def test_fuel_type_uniques_initially_none(self, vehicle):
        assert vehicle.fuel_type_uniques is None


# =============================================================
# 2. encode_fuel_type
# =============================================================

class TestEncodeFuelType:

    def test_creates_encoded_column(self, vehicle, sample_df):
        result = vehicle.encode_fuel_type(sample_df)
        assert "fuel_type_encoded" in result.columns

    def test_encoded_values_are_integers(self, vehicle, sample_df):
        result = vehicle.encode_fuel_type(sample_df)
        assert pd.api.types.is_integer_dtype(result["fuel_type_encoded"])

    def test_same_fuel_types_get_same_code(self, vehicle, sample_df):
        result = vehicle.encode_fuel_type(sample_df)
        petrol_codes = result.loc[sample_df["Type_fuel"] == "Petrol", "fuel_type_encoded"]
        assert petrol_codes.nunique() == 1

    def test_different_fuel_types_get_different_codes(self, vehicle, sample_df):
        result = vehicle.encode_fuel_type(sample_df)
        n_unique_types = sample_df["Type_fuel"].nunique()
        n_unique_codes = result["fuel_type_encoded"].nunique()
        assert n_unique_codes == n_unique_types

    def test_stores_fuel_type_uniques(self, vehicle, sample_df):
        vehicle.encode_fuel_type(sample_df)
        assert vehicle.fuel_type_uniques is not None
        assert len(vehicle.fuel_type_uniques) == sample_df["Type_fuel"].nunique()

    def test_custom_column_name(self, vehicle, sample_df):
        result = vehicle.encode_fuel_type(
            sample_df, column_to_be_created="custom_fuel"
        )
        assert "custom_fuel" in result.columns
        assert "fuel_type_encoded" not in result.columns

    def test_does_not_mutate_original_dataframe(self, vehicle, sample_df):
        original_columns = list(sample_df.columns)
        _ = vehicle.encode_fuel_type(sample_df)
        assert list(sample_df.columns) == original_columns

    def test_returns_dataframe(self, vehicle, sample_df):
        result = vehicle.encode_fuel_type(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_non_fuel_columns_unchanged(self, vehicle, sample_df):
        result = vehicle.encode_fuel_type(sample_df)
        pd.testing.assert_series_equal(result["N_doors"], sample_df["N_doors"])


# =============================================================
# 3. log_transform_vehicle_value
# =============================================================

class TestLogTransformVehicleValue:

    def test_creates_log_column(self, vehicle, sample_df):
        result = vehicle.log_transform_vehicle_value(sample_df)
        assert "Value_vehicle_log_transformed" in result.columns

    def test_log_values_are_correct(self, vehicle, sample_df):
        result = vehicle.log_transform_vehicle_value(sample_df)
        expected = np.log(sample_df["Value_vehicle"])
        pd.testing.assert_series_equal(
            result["Value_vehicle_log_transformed"], expected, check_names=False
        )

    def test_log_reduces_skewness(self, vehicle, sample_df):
        """Log-transformed values should have a smaller range than raw values."""
        result = vehicle.log_transform_vehicle_value(sample_df)
        raw_range = sample_df["Value_vehicle"].max() - sample_df["Value_vehicle"].min()
        log_range = result["Value_vehicle_log_transformed"].max() - result["Value_vehicle_log_transformed"].min()
        assert log_range < raw_range

    def test_raises_error_for_zero_values(self, vehicle):
        df = pd.DataFrame({"Value_vehicle": [0.0, 15000.0]})
        with pytest.raises(ValueError, match="zero or negative"):
            vehicle.log_transform_vehicle_value(df)

    def test_raises_error_for_negative_values(self, vehicle):
        df = pd.DataFrame({"Value_vehicle": [-100.0, 15000.0]})
        with pytest.raises(ValueError, match="zero or negative"):
            vehicle.log_transform_vehicle_value(df)

    def test_custom_column_name(self, vehicle, sample_df):
        result = vehicle.log_transform_vehicle_value(
            sample_df, column_to_be_created="custom_log"
        )
        assert "custom_log" in result.columns
        assert "Value_vehicle_log_transformed" not in result.columns

    def test_does_not_mutate_original_dataframe(self, vehicle, sample_df):
        original_values = sample_df["Value_vehicle"].copy()
        _ = vehicle.log_transform_vehicle_value(sample_df)
        pd.testing.assert_series_equal(sample_df["Value_vehicle"], original_values)

    def test_returns_dataframe(self, vehicle, sample_df):
        result = vehicle.log_transform_vehicle_value(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_value_column(self, vehicle, sample_df):
        result = vehicle.log_transform_vehicle_value(sample_df)
        pd.testing.assert_series_equal(result["Value_vehicle"], sample_df["Value_vehicle"])


# =============================================================
# 4. transform (main entry point)
# =============================================================

class TestTransform:

    def test_transform_creates_all_feature_columns(self, vehicle, sample_df):
        result = vehicle.transform(sample_df)
        assert "fuel_type_encoded" in result.columns
        assert "Value_vehicle_log_transformed" in result.columns

    def test_transform_returns_dataframe(self, vehicle, sample_df):
        result = vehicle.transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_does_not_mutate_original(self, vehicle, sample_df):
        original_columns = list(sample_df.columns)
        _ = vehicle.transform(sample_df)
        assert list(sample_df.columns) == original_columns

    def test_transform_preserves_non_transformed_columns(self, vehicle, sample_df):
        result = vehicle.transform(sample_df)
        pd.testing.assert_series_equal(result["N_doors"], sample_df["N_doors"])

    def test_transform_stores_fuel_type_uniques(self, vehicle, sample_df):
        vehicle.transform(sample_df)
        assert vehicle.fuel_type_uniques is not None

    def test_transform_log_values_are_correct(self, vehicle, sample_df):
        result = vehicle.transform(sample_df)
        expected = np.log(sample_df["Value_vehicle"])
        pd.testing.assert_series_equal(
            result["Value_vehicle_log_transformed"], expected, check_names=False
        )

    def test_transform_encoded_values_match_unique_types(self, vehicle, sample_df):
        result = vehicle.transform(sample_df)
        assert result["fuel_type_encoded"].nunique() == sample_df["Type_fuel"].nunique()

