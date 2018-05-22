import numpy as np
import pandas as pd
from sklearn.utils.validation import _num_samples
from sklearn.metrics import r2_score


__all__ = ['bootstrap_scores', 'BootstrapScorer']


def bootstrap_scores(y_true, y_pred, score_fun, n_samples):
    """
    Create Bootstrapped scores
    :param y_true: Array-like of ground truth y-values.
    :param y_pred: Array-like of predicted y-values - the same length as y_true
    :param score_fun: A function which accepts y_true and y_pred and returns a real number.
    :param n_samples: The number of times the score function is called.
    :return: An array of bootstrapped scores.
    """

    def indices_generator():
        n = _num_samples(y_true)
        for i in range(n_samples):
            yield np.random.choice(n, n, replace=True)

    # In case we were passed a pandas Series instead of an np.array or list.
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values

    return np.array([score_fun(y_true[indices], y_pred[indices]) for indices in indices_generator()])


class BootstrapScorer:

    def __init__(self, score_fun=r2_score, n_samples=100):
        """
        Construct a bootstrap scorer object
        :param score_fun: A function which accepts y_true and y_pred and returns a real number.
        :param n_samples: The number of times the score function is called.
        """
        self.score_fun = score_fun
        self.n_samples = n_samples

    def scores(self, y_true, y_pred):

        self.scores_ = bootstrap_scores(y_true, y_pred, self.score_fun, self.n_samples)
        self.description_ = pd.Series(self.scores_).describe(percentiles=np.arange(0.5, 1.0, 0.05))

        return self.scores_
