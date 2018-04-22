from sklearn.ensemble import RandomForestRegressor
from sklearn.base import TransformerMixin


__all__ = ['RandomForestTransformer']


class RandomForestTransformer(RandomForestRegressor, TransformerMixin):
    """
    Wrapper to allow the use of RandomForestRegressors as a transformer.

    If sklearn ever adds the ability to apply post-processing to predictors (regressors/classifiers)
    in a pipeline then this hacky nonsense will not be needed.
    """

    def transform(self, X, y=None):
        """
        Apply the fit random forest to transform an input array into an array of predictions.
        This is a wrapper for RandomForestRegressor.predict()

        :param X: Input array
        :param y: Not used, but required for compatability with sklearn's TransformerMixin API.
        :return: The predictions.
        """
        return self.predict(X)
