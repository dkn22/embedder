from embedder.preprocessing import categorize, size_embeddings, encode_categorical
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import OrderedDict
import warnings


checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss',
                             verbose=0, save_best_only=True,
                             save_weights_only=True,
                             mode='auto')

early_stop = EarlyStopping(monitor='val_loss', patience=10,
                           mode='min',
                           verbose=1)

def check_emb_sizes(dictionary):
    if not isinstance(dictionary, dict):
        raise TypeError('This attribute needs to contain a dictionary.')

    if not all([len(x) == 2 for x in dictionary.values()]):
        raise ValueError('Values must be tuples of length 2.')

class Base(object):

    def __init__(self, emb_sizes, model_json=None):
        self.model_json = model_json
        self._categorical_vars = emb_sizes.keys()

        check_emb_sizes(emb_sizes)
        self._emb_sizes = emb_sizes
        self.model = None

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

    def _prepare_inputs(self, X):
        numerical_vars = [x for x in X.columns
                          if x not in self._categorical_vars]

        x_inputs = []
        for col in self._categorical_vars:
            x_inputs.append(X[col].values)

        x_inputs.append(X[numerical_vars].values)

        return x_inputs

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
            # merge_idx = extractor.layers.index(
            #   extractor.get_layer('concatenate_1'))
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
        if not hasattr(self.model, 'predict'):
            raise AttributeError('Model attribute needs to be '
                                 'a Keras model with a predict method.')

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
