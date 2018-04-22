from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import pandas as pd
import numpy as np
from scipy.optimize import minimize


__all__ = ['QuantileCalibrator']


class QuantileCalibrator(BaseEstimator, TransformerMixin, RegressorMixin):
    """
    An estimator which will calibrate with respect to quantiles.
    """

    def __init__(self, quantiles=10, isotonic_fit=True, isotonic_lambda=1):
        """
        Create a quantile transformer class.
        :param quantile: Either an integer, or an array-like of floats.
        :param isotonic_fit: If true, regularize with an isotonic fit.
        :param isotonic_lambda: Lambda parameter for 3rd derivative regularization.
        """

        self.quantiles = quantiles
        self.isotonic_fit = isotonic_fit
        self.isotonic_lambda = isotonic_lambda

    # TODO: Can this be one line? If I can figure out how to add in extra rows it could be...
    #       In particular, if I could add rows with indices [-np.inf, lookup_table_.index[0].left)
    #       and [lookup_table_.index[-1].right, np.inf) I could to the lookup table I could remove
    #       these bounds checks.
    def _lookup(self, val):
        if val >= self.lookup_table_.index[-1].right:
            return self.lookup_table_.iloc[-1]
        elif val <= self.lookup_table_.index[0].left:
            return self.lookup_table_.iloc[0]
        else:
            return self.lookup_table_[val]

    @staticmethod
    def _ls_min_func(y_fit, y, lamb):
        D3_y_fit = np.diff(np.diff(np.diff(y_fit)))

        return np.inner(y - y_fit, y - y_fit) + lamb * np.inner(D3_y_fit, D3_y_fit)

    def _isotonic_fit(self, X):
        cons = ({'type': 'ineq', 'fun': lambda x: np.diff(x)})

        return minimize(self._ls_min_func,
                        x0=X,
                        args=(X, self.isotonic_lambda),
                        method='SLSQP',
                        constraints=cons).x

    def fit(self, X, y):
        """
        Fit the quantile calibration transformer.
        :param X: Array like which contains the predicted values.
        :param y: Array like which contains the ground truth values.
        :return: self
        """

        self.lookup_table_ = pd.Series(y).groupby(pd.qcut(X, self.quantiles)).mean()

        if self.isotonic_fit:
            self.lookup_table_[:] = self._isotonic_fit(self.lookup_table_.values)

        return self

    def transform(self, X, y=None):
        """
        Transform a vector via the lookup table.
        :param X: Vector to transform
        :param y: Ignored. Only included to be compatible w/ sklearn requirements for Transformers
        :return:
        """
        return np.array([self._lookup(a) for a in X])

    def predict(self, X):
        """
        Wrapper around transform. This method will be called on a
        :param X:
        :return:
        """
        return self.transform(X)
