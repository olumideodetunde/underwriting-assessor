import pytest
import pandas as pd
from src.data.splitter import split_data


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": range(100), "b": range(100, 200)})


def test_split_returns_two_dataframes(sample_df: pd.DataFrame) -> None:
    train, test = split_data(sample_df)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_split_respects_test_ratio(sample_df: pd.DataFrame) -> None:
    train, test = split_data(sample_df, test_ratio=0.3)
    assert len(test) == 30
    assert len(train) == 70


def test_split_preserves_total_rows(sample_df: pd.DataFrame) -> None:
    train, test = split_data(sample_df)
    assert len(train) + len(test) == len(sample_df)


def test_split_invalid_ratio_raises_value_error(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        split_data(sample_df, test_ratio=0.0)
    with pytest.raises(ValueError):
        split_data(sample_df, test_ratio=1.0)
    with pytest.raises(ValueError):
        split_data(sample_df, test_ratio=-0.5)


def test_split_is_reproducible(sample_df: pd.DataFrame) -> None:
    train1, test1 = split_data(sample_df, random_state=99)
    train2, test2 = split_data(sample_df, random_state=99)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)
