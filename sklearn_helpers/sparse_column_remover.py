from sklearn.base import TransformerMixin


__all__ = ['SparseColumnRemover']


class SparseColumnRemover(TransformerMixin):
    """
    A transformer which removes columns of a dataframe which are made up of too many zeros.
    """

    # TODO: Allow a parameter other than zero to be bad,
    #       or have the fitter itself determine if the column is mostly constant.
    #       Only do this if the need arises.
    def __init__(self, tolerance=0.01):
        """
        :param tolerance: We remove all columns which contain fewer than this ratio of nonzero rows.
        """
        self.tolerance = tolerance

    def fit(self, X, y=None):
        """
        Fit the transformer. Look at all columns of X and remember which have fewer than tolerance ratio of zeros.
        :param X: The data frame with inputs.
        :param y: Not used, included as a parameter for compatibility w/ sklearn
        :return: self
        """

        self.nonzeros_in_col_count_ = (X > 0).sum()
        bad_cols_bool_vec = self.nonzeros_in_col_count_ > X.shape[0] * self.tolerance
        self.columns_that_persist_ = (self.nonzeros_in_col_count_[bad_cols_bool_vec]).keys()
        self.columns_removed_count_ = X.shape[1] - len(self.columns_that_persist_)
        self.columns_that_persist_count_ = len(self.columns_that_persist_)

        return self

    # TODO: Good error message if the columns aren't in X
    def transform(self, X, y=None):
        """
        Remove columns from X which are too sparse w/r/t the trained data.
        :param X: Dataframe containing inputs
        :param y: Not used.
        :return: X with sparse columns removed.
        """
        return X[self.columns_that_persist_]