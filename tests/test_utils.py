import sys
sys.path.insert(1, '../embedder')

from embedder.preprocessing import (categorize,
pick_emb_dim, encode_categorical, replace_rare)

from sklearn.datasets import make_classification
import pandas as pd
import pytest

@pytest.fixture
def test_data():
    X, y = make_classification(n_features=20)

    X = pd.DataFrame(X).rename(columns={0: 'Categorical 1',
                                        10: 'Categorical 2'})



    X.iloc[:, 0] = X.iloc[:, 0].astype(str)
    X.iloc[:, 10] = X.iloc[:, 10].astype(str)

    for j in [0, 10]:
        X.iloc[:50, j] = ['cat_{}'.format(i) for i in range(5)] * 10
        X.iloc[50:, j] = ['cat_{}'.format(i) for i in range(5, 10)] * 10

    return X

def test_categorize(test_data):

    cat_sz = categorize(test_data)

    assert len(cat_sz) == 2
    assert cat_sz[0] == ('Categorical 1', 10)
    assert cat_sz[1] == ('Categorical 2', 10)

def test_pick_emb_dim(test_data):
    cat_sz = categorize(test_data)

    emb_sz = pick_emb_dim(cat_sz)

    assert len(emb_sz) == 2
    assert emb_sz['Categorical 1'] == (10, 5)
    assert emb_sz['Categorical 2'] == (10, 5)

def test_pick_custom_emb_dim(test_data):
    cat_sz = categorize(test_data)
    emb_sz = pick_emb_dim(cat_sz, emb_dims=[3, 4])

    assert len(emb_sz) == 2
    assert emb_sz['Categorical 1'] == (10, 3)
    assert emb_sz['Categorical 2'] == (10, 4)

def test_encode_categorical(test_data):

    df, encoders = encode_categorical(test_data)

    assert 'object' not in df.dtypes
    assert len(encoders) == 2
    assert 'Categorical 1' in encoders.keys() \
           and 'Categorical 2' in encoders.keys()

    assert all(['cat_{}'.format(i) in encoders['Categorical 1'].classes_
                for i in range(10)])
    assert all(['cat_{}'.format(i) in encoders['Categorical 2'].classes_
                for i in range(10)])

