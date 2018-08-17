import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'create_quantile_lookup_table',
    'QuantileScaler'
]

def create_quantile_lookup_table(x, n_quantiles=100):
    q = pd.Series(x).quantile(q=np.linspace(0, 1, n_quantiles + 1)).drop_duplicates().index.values

    quantile_cut = pd.qcut(x, q=q)

    ordered_quantiles = pd.Series(quantile_cut.unique()).sort_values().reset_index(drop=True).values.astype('object')

    ordered_quantiles[0] = pd.Interval(left=-np.inf, right=ordered_quantiles[0].right, closed='right')
    ordered_quantiles[-1] = pd.Interval(left=ordered_quantiles[-1].left, right=np.inf, closed='right')

    return pd.Series(data=q[:-1], index=pd.CategoricalIndex(ordered_quantiles, ordered=True))


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