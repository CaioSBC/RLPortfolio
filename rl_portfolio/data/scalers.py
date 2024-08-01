from __future__ import annotations

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)
from typing import Any


class GroupByScaler(BaseEstimator, TransformerMixin):
    """Sklearn-like scaler that scales considering groups of data.

    In the financial setting, this scale can be used to normalize a DataFrame
    with time series of multiple tickers. The scaler will fit and transform
    data for each ticker independently.
    """

    def __init__(
        self,
        by: str,
        scaler: type[
            MaxAbsScaler | MinMaxScaler | StandardScaler | RobustScaler
        ] = MaxAbsScaler,
        columns: list[str] | None = None,
        scaler_kwargs: dict[str, Any] = None,
    ) -> GroupByScaler:
        """Initializes GoupBy scaler.

        Args:
            by: Name of column that will be used to group.
            scaler: Scikit-learn scaler class to be used.
            columns: List of columns that will be scaled.
            scaler_kwargs: Keyword arguments for chosen scaler.
        """
        self.scalers = {}  # dictionary with scalers
        self.by = by
        self.scaler = scaler
        self.columns = columns
        self.scaler_kwargs = {} if scaler_kwargs is None else scaler_kwargs

    def fit(self, X: pd.DataFrame, y: Any = None) -> GroupByScaler:
        """Fits the scaler to input data.

        Args:
            X: DataFrame to fit.
            y: Not used.

        Returns:
            Fitted GroupBy scaler.
        """
        # if columns aren't specified, considered all numeric columns
        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object"]).columns
        # fit one scaler for each group
        for value in X[self.by].unique():
            X_group = X.loc[X[self.by] == value, self.columns]
            self.scalers[value] = self.scaler(**self.scaler_kwargs).fit(X_group)
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Transforms unscaled data.

        Args:
            X: DataFrame to transform.
            y: Not used.

        Returns:
            Transformed DataFrame.
        """
        # apply scaler for each group
        X = X.copy()
        for value in X[self.by].unique():
            select_mask = X[self.by] == value
            X.loc[select_mask, self.columns] = self.scalers[value].transform(
                X.loc[select_mask, self.columns]
            )
        return X
