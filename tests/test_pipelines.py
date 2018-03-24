from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from embedder.regression import Embedder
from embedder.preprocessing import categorize, pick_emb_dim, encode_categorical
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def test_data():
    X, y = make_regression(n_samples=10000, n_features=20)

    X = pd.DataFrame(X).rename(columns={0: 'Categorical 1',
                                        10: 'Categorical 2'})



    X.iloc[:, 0] = X.iloc[:, 0].astype(str)
    X.iloc[:, 10] = X.iloc[:, 10].astype(str)

    for j in [0, 10]:
        X.iloc[:5000, j] = ['cat_{}'.format(i) for i in range(500)] * 10
        X.iloc[5000:, j] = ['cat_{}'.format(i) for i in range(500, 1000)] * 10

    return X, pd.Series(y)

def test_pipeline(test_data):
    X, y = test_data
    cat_sz = categorize(X)
    emb_sz = pick_emb_dim(cat_sz)
    X_encoded, encoders = encode_categorical(X)

    pipeline = Pipeline(
        [('embedding', Embedder(emb_sz)),
         ('randomforest', RandomForestRegressor())]
    )

    pipeline.fit(X_encoded, y)
    preds = pipeline.predict(X_encoded)

    assert len(preds) == 10000
    assert not np.isinf(preds).any()
    assert not np.isnan(preds).any()

