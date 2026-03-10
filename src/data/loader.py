import pandas as pd


def load_csv(
    path: str,
    delimiter: str = ";",
    na_values: list[str] | None = None,
    parse_dates: list[str] | None = None,
    date_format: str = "%d/%m/%Y",
    dayfirst: bool = True,
) -> pd.DataFrame:
    if na_values is None:
        na_values = ["NA", ""]
    df = pd.read_csv(
        path,
        delimiter=delimiter,
        na_values=na_values,
        parse_dates=parse_dates if parse_dates else False,
        date_format=date_format if parse_dates else None,
        dayfirst=dayfirst,
    )
    return df


def aggregate_claims(
    claims: pd.DataFrame,
    policy: pd.DataFrame,
    group_columns: list[str],
    agg_column: str,
    agg_method: str = "count",
    target_name: str = "claims_frequency",
) -> pd.DataFrame:
    """Aggregate claims to one row per policy-period and fill gaps with 0.

    Steps:
    1. Build the full index of (ID, year) combos from the policy table
       so every policy-period is represented.
    2. Group + aggregate the claims table.
    3. Left-join and fill NaN with 0 (no claim = 0 frequency).
    """
    full_index = policy[group_columns].drop_duplicates()

    claims_agg = (
        claims
        .groupby(group_columns)
        .agg({agg_column: agg_method})
        .rename(columns={agg_column: target_name})
        .reset_index()
    )

    merged = (
        full_index
        .merge(claims_agg, on=group_columns, how="left")
        .assign(**{target_name: lambda df: df[target_name].fillna(0)})
    )

    return merged


def build_modelling_dataset(
    policy_path: str,
    claims_path: str,
    group_columns: list[str],
    agg_column: str,
    merge_columns: list[str],
    agg_method: str = "count",
    target_name: str = "claims_frequency",
    delimiter: str = ";",
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """End-to-end: load → aggregate → merge → return one modelling-ready DataFrame."""
    policy = load_csv(policy_path, delimiter=delimiter, parse_dates=parse_dates)
    claims = load_csv(claims_path, delimiter=delimiter)

    claims_agg = aggregate_claims(
        claims=claims,
        policy=policy,
        group_columns=group_columns,
        agg_column=agg_column,
        agg_method=agg_method,
        target_name=target_name,
    )

    dataset = pd.merge(
        left=policy,
        right=claims_agg,
        on=merge_columns,
        how="left",
    )

    return dataset

