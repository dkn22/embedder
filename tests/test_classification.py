from sklearn.datasets import make_classification
from embedder.classification import Embedder
from embedder.preprocessing import categorize, pick_emb_dim, encode_categorical
import pandas as pd
import pytest

@pytest.fixture
def test_data():
    X, y = make_classification(n_samples=10000, n_features=20)

    X = pd.DataFrame(X).rename(columns={0: 'Categorical 1',
                                        10: 'Categorical 2'})



    X.iloc[:, 0] = X.iloc[:, 0].astype(str)
    X.iloc[:, 10] = X.iloc[:, 10].astype(str)

    for j in [0, 10]:
        X.iloc[:5000, j] = ['cat_{}'.format(i) for i in range(500)] * 10
        X.iloc[5000:, j] = ['cat_{}'.format(i) for i in range(500, 1000)] * 10

    return X, pd.Series(y)

def test_fit_predict(test_data):
    X, y = test_data
    cat_sz = categorize(X)
    emb_sz = pick_emb_dim(cat_sz)
    X, encoders = encode_categorical(X)

    embedder = Embedder(emb_sz)
    embedder.fit(X, y, epochs=1)

    preds = embedder.predict(X[:100])

    assert len(preds) == 100
    assert all(preds <= 1) and all(preds >= 0)

def test_transform(test_data):
    X, y = test_data
    cat_sz = categorize(X)
    emb_sz = pick_emb_dim(cat_sz, max_dim=50)
    X, encoders = encode_categorical(X)

    embedder = Embedder(emb_sz)
    embedder.fit(X, y, epochs=1)

    transformed = embedder.transform(X)

    assert transformed.shape == (10000, 18 + 50 + 50)













