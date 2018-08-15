[![Build Status](https://travis-ci.org/dkn22/embedder.svg?branch=master)](https://travis-ci.org/dkn22/embedder)
[![Twitter Follow](https://img.shields.io/twitter/follow/espadrine.svg?style=social&logo=twitter&label=Follow)](https://twitter.com/dat_nguyen)

#Fixed issue with as_df=True option in transform 
# Overview
embedder is a small utility tool meant to simplify training, pre-processing and extraction of entity embeddings through neural networks.

# Installation

It is recommended to create a virtual environment first.

To install a package, clone the repository and run:
```bash
python setup.py install
```
# Example – training
```python
rossman = pd.read_csv('rossman.csv')
y = rossman['Sales']
X = rossman.drop('Sales', axis=1)
cat_vars = categorize(rossman)
embedding_dict = pick_emb_dim(cat_vars, max_dim=50)
X_encoded, encoders = encode_categorical(X)
embedder = Embedder(embedding_dict, model_json=None)
embedder.fit(X_encoded, y)
```

Let’s examine what embedder is trying to do here. 

1. It determines the categorical variables in the data by examining data types of the columns in the pandas DataFrame and the number of unique categories in each variable. 
2. Then, it prepares a dictionary of variables to be embedded and the dimensionality of the embeddings. Recall that an embedding is a fixed-length vector representation of a category. Here, embedder determines embedding sizes using a rule of thumb: it simply takes the minimum of the half of the number of unique categories or the maximum dimensionality allowed, which is passed as an argument. Those defaults have worked very well in my experience. However, nothing prevents a user from passing different dictionary — it only has to be of the same format.
3. The categorical variables are encoded using integer encoding, as this is the data type that Keras, and any other major deep learning framework, would expect. The encoders that map categories to integers are also returned — this may become useful to later assess learnt embeddings, e.g. by labelling them. Note that these pre-processing steps are only meant to simplify the process of preparing the data prior to training a neural network, but are not mandatory.
4. Finally, the main class is instantiated and a neural network is fit on the pre-processed data. Two things to point out — by default, embedder will train a feedforward network with two hidden layers, which is a sensible default. Of course, it may not be optimal for all possible applications. The desired architecture can be passed as a json at class instantiation. Second, by default on a regression task a mean squared error loss function will be used (and cross-entropy loss for classification tasks)— again, a sensible default for vanilla applications that embedder aims to simplify.

# References
* [Guo and Berkhahn (2016): Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737)

# Contribution
Any contributions are welcome and you can [email](mailto:dat.nguyen@cantab.net) me for troubleshooting.
