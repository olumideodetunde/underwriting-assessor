import pandas as pd

from src.feature.base import BaseFeatureTransformer
from src.feature.driver import Driver
from src.feature.vehicle import Vehicle


class FittedFeaturePipeline(BaseFeatureTransformer):
    """Compose the Vehicle and Driver transformers into one fit/transform unit.

    fit() learns state on the training split only (via Vehicle.fit); transform()
    applies the learned pipeline to any split, in the same order used by training:
    Vehicle first, then Driver. Transform is guarded against being called before fit.
    """

    def __init__(self) -> None:
        self.vehicle = Vehicle()
        self.driver = Driver()
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FittedFeaturePipeline":
        # Learn state from the training data only. Driver.fit is a no-op but keeps
        # the interface symmetric across both transformers.
        self.vehicle.fit(df)
        self.driver.fit(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                "FittedFeaturePipeline must be fitted before transform; call fit() first."
            )
        out = self.vehicle.transform(df)
        out = self.driver.transform(out)
        return out
