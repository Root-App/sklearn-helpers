import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'create_quantile_lookup_table',
    'QuantileScaler'
]


def round_to_n(x, n=8):
    """
    Round a number to n significant digits.

    e.g.

    round_to_n(12345, 2) -> 12000
    round_to_n(-.00897, 1) -> -.009

    :param x: Number to round
    :param n: number of significant digits
    :return: The rounded number
    """
    m = n - 1 - int(np.floor(np.log10(abs(x) + .01)))

    return round(x, m)


def create_quantile_lookup_table(x, n_quantiles=100):
    x = x.apply(round_to_n)

    q = pd.Series(x).quantile(q=np.linspace(0, 1, n_quantiles + 1))

    q = q.drop_duplicates()

    q[0] = -np.inf
    q[-1] = np.inf

    intervals = [pd.Interval(left=a, right=b, closed='right') for a, b in zip(q.iloc[:-1], q.iloc[1:])]

    return pd.Series(data=q.index[:-1], index=pd.IntervalIndex(intervals))


class QuantileScaler(BaseEstimator, TransformerMixin):
    """
    Perform a monotone transformation on a feature to map into buckets of equal size.
    """

    def __init__(self, n_quantiles=10):
        """
        Initialize the transformer.
        :param n_quantiles: Number of quantiles to map to
        """
        self.n_quantiles = n_quantiles

    def fit(self, X, y=None):
        n_unique_values = len(np.unique(X))

        if self.n_quantiles > n_unique_values:
            self.n_quantiles = n_unique_values

        self.lookup_table_ = create_quantile_lookup_table(X, self.n_quantiles)

        return self

    def transform(self, X):
        # Note: this is kinda slow.
        # TODO: There are some speedups available for this but it's not imperative.
        return pd.Series(X).apply(lambda a: self.lookup_table_[a]).values