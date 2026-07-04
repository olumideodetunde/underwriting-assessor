import pytest
import pandas as pd

from src.feature.base import BaseFeatureTransformer
from src.feature.driver import Driver
from src.feature.pipeline import FittedFeaturePipeline
from src.feature.vehicle import Vehicle


# --- Fixtures ---

@pytest.fixture
def pipeline():
    return FittedFeaturePipeline()


@pytest.fixture
def sample_df():
    """A frame carrying the columns both transformers need.

    Vehicle reads Type_fuel / Value_vehicle; Driver reads the Date* columns
    (dd/mm/yyyy). N_doors is a passthrough column used to assert non-mutation.
    """
    return pd.DataFrame({
        "Type_fuel": ["Petrol", "Diesel", "Petrol", "LPG", "Diesel"],
        "Value_vehicle": [15000.0, 25000.0, 8000.0, 12000.0, 30000.0],
        "Date_last_renewal": ["01/06/2018", "15/03/2019", "20/09/2017", "10/01/2020", "05/12/2016"],
        "Date_birth": ["01/06/1980", "15/03/1975", "20/09/1990", "10/01/1968", "05/12/1985"],
        "Date_driving_licence": ["01/06/2000", "15/03/1995", "20/09/2010", "10/01/1988", "05/12/2004"],
        "N_doors": [4, 4, 2, 4, 4],
    })


@pytest.fixture
def fitted_pipeline(sample_df):
    """A pipeline that has learned its state from sample_df."""
    return FittedFeaturePipeline().fit(sample_df)


VEHICLE_COLUMNS = ["fuel_type_encoded", "Value_vehicle_log_transformed"]
DRIVER_COLUMNS = [
    "driver_age_at_contract_inception",
    "driver_experience_age",
    "driver_age_experience_age_diff",
    "driver_age_experience_ratio_proxy_for_driving_experience",
]


# =============================================================
# 1. Instantiation
# =============================================================

class TestPipelineInstantiation:

    def test_pipeline_is_instance_of_base(self, pipeline):
        assert isinstance(pipeline, BaseFeatureTransformer)

    def test_pipeline_instantiates_successfully(self, pipeline):
        assert isinstance(pipeline, FittedFeaturePipeline)

    def test_owns_a_vehicle_and_driver(self, pipeline):
        assert isinstance(pipeline.vehicle, Vehicle)
        assert isinstance(pipeline.driver, Driver)

    def test_not_fitted_initially(self, pipeline):
        assert pipeline._fitted is False


# =============================================================
# 2. fit / transform contract
# =============================================================

class TestFitTransformContract:

    def test_fit_returns_self(self, pipeline, sample_df):
        assert pipeline.fit(sample_df) is pipeline

    def test_fit_sets_fitted_flag(self, pipeline, sample_df):
        pipeline.fit(sample_df)
        assert pipeline._fitted is True

    def test_fit_fits_the_underlying_vehicle(self, pipeline, sample_df):
        pipeline.fit(sample_df)
        assert pipeline.vehicle.fuel_type_uniques_ is not None

    def test_transform_before_fit_raises(self, pipeline, sample_df):
        with pytest.raises(RuntimeError, match="fitted before transform"):
            pipeline.transform(sample_df)

    def test_codes_consistent_across_reordered_data(self):
        """Leakage regression: a fuel type must map to the same code on the fitted
        train frame and on a differently-ordered test frame."""
        train = pd.DataFrame({
            "Type_fuel": ["Petrol", "Diesel", "LPG"],
            "Value_vehicle": [10000.0, 20000.0, 30000.0],
            "Date_last_renewal": ["01/06/2018", "15/03/2019", "20/09/2017"],
            "Date_birth": ["01/06/1980", "15/03/1975", "20/09/1990"],
            "Date_driving_licence": ["01/06/2000", "15/03/1995", "20/09/2010"],
        })
        # Test set with a different first-appearance order.
        test = pd.DataFrame({
            "Type_fuel": ["Diesel", "Petrol", "LPG"],
            "Value_vehicle": [20000.0, 10000.0, 30000.0],
            "Date_last_renewal": ["15/03/2019", "01/06/2018", "20/09/2017"],
            "Date_birth": ["15/03/1975", "01/06/1980", "20/09/1990"],
            "Date_driving_licence": ["15/03/1995", "01/06/2000", "20/09/2010"],
        })

        pipeline = FittedFeaturePipeline().fit(train)
        train_out = pipeline.transform(train)
        test_out = pipeline.transform(test)

        train_map = dict(zip(train["Type_fuel"], train_out["fuel_type_encoded"]))
        test_map = dict(zip(test["Type_fuel"], test_out["fuel_type_encoded"]))

        assert train_map == test_map
        assert train_map["Petrol"] == 0  # learned first -> always code 0


# =============================================================
# 3. transform (main entry point)
# =============================================================

class TestTransform:

    def test_transform_returns_dataframe(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_creates_vehicle_columns(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        for col in VEHICLE_COLUMNS:
            assert col in result.columns

    def test_transform_creates_driver_columns(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        for col in DRIVER_COLUMNS:
            assert col in result.columns

    def test_transform_does_not_mutate_original(self, fitted_pipeline, sample_df):
        original_columns = list(sample_df.columns)
        _ = fitted_pipeline.transform(sample_df)
        assert list(sample_df.columns) == original_columns

    def test_transform_preserves_passthrough_columns(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        pd.testing.assert_series_equal(result["N_doors"], sample_df["N_doors"])
