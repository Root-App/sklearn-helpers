import numpy as np

from sklearn.utils.validation import _num_samples
from sklearn.model_selection import RandomizedSearchCV


__all__ = ['RandomizedSearchOOB']


class _OneFold:
    """
    A cross validation generator that generates a single fold containing all examples in both train and test.
    """
    def split(self, X=None, y=None, groups=None):
        return [(np.array(np.arange(_num_samples(X))),
                 np.array(np.arange(_num_samples(X))))]

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


class RandomizedSearchOOB(RandomizedSearchCV):
    """
    Perform a random hyper-parameter search. Instead of cross validating the results using a hold-out set,
    validate using an out-of-bag prediction. This is only possible with estimators that have such sets, e.g.
    random forests.
    """

    def __init__(self,
                 estimator,
                 param_distributions,
                 n_iter=10,
                 scoring=None,
                 fit_params=None,
                 n_jobs=1,
                 iid=True,
                 refit=True,
                 verbose=0,
                 pre_dispatch='2*n_jobs',
                 random_state=None,
                 error_score='raise'):

        if not hasattr(estimator, 'oob_score'):
            raise ValueError("RandomizedSearchOOB requires the ability to use out-of-bag predictions. "
                             "You passed the estimator {}, which does not have the attribute 'oob_score. ".
                             format(estimator))

        if not estimator.oob_score:
            raise ValueError("RandomizedSearchOOB requires the ability to use out-of-bag predictions. "
                             "You passed the estimator {}, which 'oob_score == False. ".format(estimator) +
                             "Set oob_score = True.")

        cv = _OneFold()
        return_train_score = False

        super().__init__(estimator,
                         param_distributions,
                         n_iter,
                         scoring,
                         fit_params,
                         n_jobs,
                         iid,
                         refit,
                         cv,
                         verbose,
                         pre_dispatch,
                         random_state,
                         error_score,
                         return_train_score)