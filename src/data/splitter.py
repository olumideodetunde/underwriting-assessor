import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    train, test = train_test_split(
        df,
        test_size=test_ratio,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,)
    return train.reset_index(drop=True), test.reset_index(drop=True)

