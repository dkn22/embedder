from embedder.base import Base
from keras.models import Sequential, Model, model_from_json
from keras.layers import (Dense, Dropout, Embedding,
                          Activation, Input, concatenate, Reshape, Flatten)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam

# from embedder.metrics import precision, recall

class Embedder(Base):

    def __init__(self, emb_sizes, model_json=None):

        super(Embedder, self).__init__(emb_sizes, model_json)

    def fit(self, X, y,
            batch_size=256, epochs=100,
            checkpoint=None,
            early_stop=None):

        nnet = self._create_model(X, model_json=self.model_json)

        nnet.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

        callbacks = [checkpoint, early_stop]

        x_inputs_list = self._prepare_inputs(X)

        nnet.fit(x_inputs_list, y.values, batch_size=batch_size,
                 epochs=epochs,
                 callbacks=[cb for cb in callbacks if cb is not None],
                 validation_split=0.2)

        self.model = nnet

        return self

    def fit_transform(self, X, y,
                      batch_size=256, epochs=100,
                      checkpoint=None,
                      early_stop=None
                      ):
        self.fit(X, y, batch_size, epochs,
                 checkpoint, early_stop)

        return self.transform(X)

    def _default_nnet(self, X):

        emb_sz = self.emb_sizes
        numerical_vars = [x for x in X.columns
                          if x not in self._categorical_vars]

        inputs = []
        flatten_layers = []

        for var, sz in emb_sz.items():
            input_c = Input(shape=(1,), dtype='int32')
            embed_c = Embedding(*sz, input_length=1)(input_c)
            # embed_c = Dropout(0.25)(embed_c)
            flatten_c = Flatten()(embed_c)

            inputs.append(input_c)
            flatten_layers.append(flatten_c)

        input_num = Input(shape=(len(numerical_vars),), dtype='float32')
        flatten_layers.append(input_num)
        inputs.append(input_num)

        flatten = concatenate(flatten_layers, axis=-1)

        fc1 = Dense(1000, kernel_initializer='normal')(flatten)
        fc1 = Activation('relu')(fc1)
        # fc1 = BatchNormalization(fc1)
        # fc1 = Dropout(0.75)(fc1)

        fc2 = Dense(500, kernel_initializer='normal')(fc1)
        fc2 = Activation('relu')(fc2)
        # fc2 = BatchNormalization(fc2)
        # fc2 = Dropout(0.5)(fc2)

        output = Dense(1, activation='sigmoid')(fc2)
        nnet = Model(inputs=inputs, outputs=output)

        return nnet
