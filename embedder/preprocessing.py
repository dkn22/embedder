from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
# from sklearn.preprocessing.label import _check_numpy_unicode_bug
import numpy as np


def categorize(X):
    cat_sz = [(col, X[col].unique().shape[0]) for col in X.columns
              if X[col].dtype == 'object']

    return cat_sz


def size_embeddings(cat_sz,
                    max_dim=50,
                    include_unseen=False):
    if include_unseen:
        emb_sz = {var: (c + 1, min(max_dim, (c + 1) // 2)) for var, c in cat_sz}
    else:
        emb_sz = {var: (c, min(max_dim, (c + 1) // 2)) for var, c in cat_sz}

    return emb_sz

    # if not isinstance(emb_sizes, dict):
    #     raise TypeError('Custom embedding sizes need to be '
    #                     'provided as a dictionary of column name'
    #                     ':embedding dimension pairs.')


def encode_categorical(X,
                       categorical_vars=None,
                       copy=True):
    df = X.copy() if copy else X
    encoders = {}

    if categorical_vars is None:
        categorical_vars = [col for col in df.columns
                            if df[col].dtype == 'object']

    for var in categorical_vars:
        encoders[var] = SafeLabelEncoder()
        encoders[var].fit(df[var])

        df.loc[:, var] = encoders[var].transform(df.loc[:, var])

    return df, encoders

def replace_rare(X, threshold=10, code='rare',
                 categorical_vars=None,
                 copy=True):

    df = X.copy() if copy else X

    if categorical_vars is None:
        categorical_vars = [col for col in df.columns
                            if df[col].dtype == 'object']

    for col in categorical_vars:
        counts = df[col].value_counts()
        rare_values = counts[counts < threshold].index
        df.loc[:, col] = df[col].map({val: code if val in rare_values
                                      else val for val in df[col].unique()})

    return df

class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value of 99999

    Attributes
    ----------

    classes_ : the classes that are encoded
    """

    def transform(self, y):

        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        # classes = np.unique(y)
        # _check_numpy_unicode_bug(classes)

        unseen = len(self.classes_)

        # if len(classes) >= unseen:
        #     raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        e = np.array([
                     np.searchsorted(self.classes_, x) if x in self.classes_ else unseen
                     for x in y
                     ])

        if unseen in e:
            self.classes_ = np.array(self.classes_.tolist() + ['unseen'])

        return e

