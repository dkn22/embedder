from sklearn.preprocessing import LabelEncoder

def categorize(X):
    cat_sz = [(col, X[col].unique().shape[0]) for col in X.columns
              if X[col].dtype == 'object']

    return cat_sz


def size_embeddings(cat_sz,
                    max_dim=50,
                    emb_sizes=None
                    ):
    if emb_sizes is None:
        emb_sz = {var: (c, min(max_dim, (c + 1) // 2)) for var, c in cat_sz}

        return emb_sz

    if not isinstance(emb_sizes, dict):
        raise TypeError('Custom embedding sizes need to be '
                        'provided as a dictionary of column name'
                        ':embedding dimension pairs.')




def encode_categorical(X, categorical_vars=None, copy=True):
    df = X.copy() if copy else X
    encoders = {}

    if categorical_vars is None:
        categorical_vars = [col for col in df.columns
                            if df[col].dtype == 'object']

    for var in categorical_vars:
        encoders[var] = LabelEncoder()
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

