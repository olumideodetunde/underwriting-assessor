
import pytest
from src.model.factory import select_training_algorithm


class TestSelectTrainingAlgorithm:

    def test_returns_valid_model_instance(self):
        """Known algorithm + valid params should return a fitted-ready model."""
        model = select_training_algorithm("gamma_regressor", {"alpha": 10, "solver": "newton-cholesky"})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_unknown_algorithm_raises_key_error(self):
        """Typo in config should fail loudly, not silently return None."""
        with pytest.raises(KeyError):
            select_training_algorithm("xgboooost", {})

    def test_bad_params_raises_type_error(self):
        """Valid algorithm but wrong params should fail at instantiation, not at .fit() time."""
        with pytest.raises(TypeError):
            select_training_algorithm("gamma_regressor", {"not_a_real_param": 999})
