import numpy as np
import pandas as pd
from sklearn.utils.validation import _num_samples


def bootstrap_score(y_true, y_pred, score_fun, n_samples):
    """
    Create Bootstrapped scores
    :param y_true: Array-like of ground truth y-values.
    :param y_pred: Array-like of predicted y-values - the same length as y_true
    :param score_fun: A function which accepts y_true and y_pred and returns a real number.
    :param n_samples: The number of times the score function is called.
    :return: An array of bootstrapped scores.
    """

    def next_indices():
        n = _num_samples(y_true)
        for i in range(n_samples):
            yield np.random.choice(n, n, replace=True)

    return np.array([score_fun(y_true[indices], y_pred[indices]) for indices in next_indices()])


class BootstrapScore:

    def __init__(self, score_fun, n_samples=100):
        self.score_fun = score_fun
        self.n_samples = n_samples

    def scores(self, y_true, y_pred):

        self.scores_ = bootstrap_score(y_true, y_pred, self.score_fun, self.n_samples)
        self.description_ = pd.Series(self.scores_).describe(percentiles=np.arange(0.5, 1.0, 0.05))

        return self.scores_