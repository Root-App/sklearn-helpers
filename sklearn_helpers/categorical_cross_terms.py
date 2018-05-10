from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd


__all__ = ['CategoricalCrossTermTransformer',
           'categorical_cross_term_transform']


class CategoricalCrossTermTransformer(BaseEstimator, TransformerMixin):
    """
    Class which handles creation of compound categorical variables from simpler ones.
    E.g. If columns 'age' and 'gender' exist, this will create a new column 'age_gender'.

    Note: This class will likely no longer be needed once CategoricalTransformer is added to scikit-learn in
    version 0.20.0. But that's not out yet, so we will keep it here until then.
    """

    VALID_BEHAVIORS = ['append', 'replace', 'series']

    def __init__(self,
                 columns=None,
                 new_column_name=None,
                 column_name_joiner='_',
                 feature_name_joiner='_',
                 behavior='append'):
        """
        Take several categorical variables and create a new one by combining them.
        :param columns: Names of the columns which contain the categorical variables.
               If 'None' then all columns are used. Default: None.
        :param new_column_name: New column name. If None, instead the name will be
               a combination of the values passed in columns. Default: None.
        :param column_name_joiner: string used to form the new column name. Default: '_'.
        :param feature_name_joiner: string used to form the new feature names. Default: '_'.
        :param behavior: One of 'append', 'replace', or 'series'. Default: 'append'
               Determines what is returned. If 'append' then the new column is appended to the
               input data frame. If 'replace', then the new column is appended and the input columns are
               removed. If 'series', the only the new column is returned as a pandas.Series. Note that
               append and replace both involve making a copy of the input data frame, which may not be desired
               if it is very large.
        """

        if behavior not in self.VALID_BEHAVIORS:
            raise ValueError('Invalid behavior. Must be one of: {}. Passed: {}'.format(
                self.VALID_BEHAVIORS, behavior))

        self.behavior = behavior
        self.columns = columns
        self.column_name_joiner = column_name_joiner

        if new_column_name is None:
            self.new_column_name = column_name_joiner.join(columns)
        else:
            self.new_column_name = new_column_name

        self.feature_name_joiner = feature_name_joiner

    # For compatibility with Pipelines
    def fit(self, X=None, y=None):
        pass

    def transform(self, X, y=None):
        """
        Apply the transform.
        :param X: Input data frame.
        :param y: Not uses, but included for compatibility with sklearn.
        :return: The transformed data.
        """

        new_column = categorical_cross_term_transform(X, self.columns, self.feature_name_joiner)

        if self.behavior == 'series':
            result = pd.Series(data=new_column, index=X.index, name=self.new_column_name)

        else:
            result = X.copy()
            result[self.new_column_name] = new_column

            if self.behavior == 'replace':
                result.drop(self.columns, axis=1, inplace=True)

        return result


def categorical_cross_term_transform(X, columns=None, feature_name_joiner='_'):
    """
    Create categorical cross term array.
    :param X: Input data
    :param columns: names of the columns which are to be combined. If None,
           all columns are used. Default: None.
    :param feature_name_joiner: string used to combine feature names. Default: '_'.
    :return: Numpy.array which contains the cross terms.
    """

    if columns is None:
        columns = X.columns

    zip_obj = zip(*[X[name].astype(str) for name in columns])

    return np.array([feature_name_joiner.join(ll) for ll in zip_obj])
