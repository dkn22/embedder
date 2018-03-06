from embedder.preprocessing import categorize, size_embeddings, encode_categorical
from keras.models import model_from_json
from collections import OrderedDict
from copy import deepcopy
import warnings

def check_emb_sizes(dictionary):
    if not isinstance(dictionary, dict):
        raise TypeError('This attribute needs to contain a dictionary.')

    if not all([len(x) == 2 for x in dictionary.values()]):
        raise ValueError('Values must be tuples of length 2.')

class Base(object):

    def __init__(self, emb_sizes, model_json=None):
        self.model_json = model_json
        # self.max_emb_dim = max_emb_dim

        self._categorical_vars = emb_sizes.keys()
        # self._numerical_vars = None

        check_emb_sizes(emb_sizes)
        self._emb_sizes = emb_sizes
        self.model = None
        # self.encoders_ = None

    @property
    def emb_sizes(self):
        return OrderedDict(self._emb_sizes)

    @emb_sizes.setter
    def emb_sizes(self, dictionary):
        check_emb_sizes(dictionary)

        if self._categorical_vars is None \
                or len(dictionary.keys()) != len(self._categorical_vars):
            raise ValueError('Number of categorical variables'
                             'is undefined yet or inconsistent.')

        self._emb_sizes = dictionary

    def __repr__(self):
        params = {'emb_sizes': self.emb_sizes,
                  'model': self.model
                  }

        return '%s(%r)' % (self.__class__.__name__, params)

    # def __getattribute__(self, item):
    #     if item.startswith('fit'):
    #         if self._emb_sizes is None:
    #             warnings.warn('Embedding sizes are not available.')
    #
    #     return object.__getattribute__(self, item)

    def _prepare_inputs(self, X):
        # if self._emb_sizes is None or 'object' in X.dtypes:
        #     # warnings.warn('X contains object types. Will attempt to encode to integers.')
        #     X = self._categorize(X, encode_categorical=True)
        numerical_vars = [x for x in X.columns
                          if x not in self._categorical_vars]

        x_inputs = []
        for col in self._categorical_vars:
            x_inputs.append(X[col].values)

        x_inputs.append(X[numerical_vars].values)

        return x_inputs

    # def _categorize(self, X, encode_categorical=True):
    #
    #     max_emb_dim = self.max_emb_dim
    #
    #     cat_sz = categorize(X)
    #     emb_sz = size_embeddings(cat_sz, max_dim=max_emb_dim)
    #
    #     self._emb_sizes = emb_sz
    #     self._categorical_vars = emb_sz.keys()
    #     self._numerical_vars = [x for x in X.columns
    #                             if x not in emb_sz.keys()]
    #
    #     if encode_categorical:
    #         X, encoders = encode_categorical(X, self._categorical_vars)
    #         self.encoders_ = encoders
    #
    #     return X

    def _create_model(self, X,
                      model_json=None):

        if model_json is None:
            if hasattr(self, '_default_nnet'):
                nnet = self._default_nnet(X)
            else:
                raise ValueError('No model architecture provided.')
        else:
            nnet = model_from_json(model_json)

        self.model_json = nnet.to_json()

        return nnet

    def transform(self, X,
                  as_df=False,
                  merge_idx=None):
        if self.model is None:
            raise AttributeError('No trained model available.')

            # create a copy of the model
        learned_weights = self.model.get_weights()
        extractor = model_from_json(self.model.to_json())
        extractor.set_weights(learned_weights)

        if merge_idx is None:
            # merge_idx = extractor.layers.index(extractor.get_layer('concatenate_1'))
            merge_idx = [idx for idx, layer in enumerate(extractor.layers)
                         if 'Concatenate' in str(layer)][0]

        extractor.layers = extractor.layers[:merge_idx + 1]
        extractor.outputs = [extractor.layers[-1].output]
        extractor.layers[-1].outbound_nodes = []

        x_inputs_list = self._prepare_inputs(X)

        embedded = extractor.predict(x_inputs_list)

        if as_df:
            sizes = [(var, sz[1]) for var, sz in self.emb_sizes]
            numerical_vars = [x for x in X.columns
                              if x not in self._categorical_vars]
            names = [var + '_{}'.format(x) for x in range(emb_dim)
                     for var, emb_dim in sizes]

            names += numerical_vars

            embedded = pd.DataFrame(embedded, columns=names)

        return embedded

    def predict(self, X_test):

        # X_test = X.copy()

        if not hasattr(self.model, 'predict'):
            raise AttributeError('Model attribute needs to be '
                                 'a Keras model with a predict method.')

        # if self.encoders_ is not None:
        #     for var, encoder in self.encoders_.items():
        #         X_test.loc[:, var] = encoder.transform(X_test.loc[:, var])

        test_inputs_list = self._prepare_inputs(X_test)
        preds = self.model.predict(test_inputs_list)

        return preds

    def get_embeddings(self):
        if self.model is None:
            raise AttributeError('No trained model available.')

        nnet = self.model

        emb_sizes = self._emb_sizes
        emb_layers = [x for x in nnet.layers if 'Embedding' in str(x)]

        if self._emb_sizes is None:
            try:
                warnings.warn('No embedding names found.'
                              'Will revert to generic names.')

                emb_sizes = {'var_{}'.format(x): (l.input_shape, l.output_shape)
                             for x, l in zip(range(len(emb_layers)), emb_layers)}
            except Exception:
                raise AttributeError('No specified embeddings available '
                                     'and failed to infer.')

        embs = list(map(lambda x: x.get_weights()[0], emb_layers))
        embeddings = {var: emb for var, emb in zip(emb_sizes.keys(), embs)}

        return embeddings

    def fit(self, X, y):
        raise NotImplementedError('fit method needs to be implemented.')

    # def fit_transform(self, X, y):
    #     raise NotImplementedError(
    #         'fit_transform method needs to be implemented.')
