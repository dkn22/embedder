from embedder.base import Base
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model, model_from_json
from keras.layers import (Dense, Dropout, Embedding,
                          Activation, Input, concatenate, Reshape, Flatten)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam

# from embedder.metrics import precision, recall


class Embedder(Base):

    def __init__(self, max_emb_dim=50, model_json=None):

        super(Embedder, self).__init__(max_emb_dim, model_json)

    def fit(self, X, y,
            batch_size=256, epochs=100,
            save_checkpoint=True,
            early_stopping=True,
            weights_filename=None):

        X = self._categorize(X, encode=True)
        nnet = self._create_model(X, model_json=self.model_json)

        filename = weights_filename \
            if weights_filename is not None else 'weights.hdf5'

        nnet.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filename, monitor='val_loss',
                                     verbose=0, save_best_only=True,
                                     save_weights_only=True,
                                     mode='auto') if save_checkpoint else None

        early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                   mode='min',
                                   verbose=1) if early_stopping else None

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
                      save_checkpoint=True,
                      early_stopping=True,
                      weights_filename=None
                      ):

        self.fit(X, y, batch_size, epochs,
                 save_checkpoint, early_stopping, weights_filename)

        return self.transform(X)

    def _default_nnet(self, X):

        if self._emb_sizes is None:
            self._categorize(X, encode=False)

        emb_sz = self.emb_sizes

        inputs = []
        flatten_layers = []

        for var, sz in emb_sz.items():
            input_c = Input(shape=(1,), dtype='int32')
            embed_c = Embedding(*sz, input_length=1)(input_c)
            # embed_c = Dropout(0.25)(embed_c)
            flatten_c = Flatten()(embed_c)

            inputs.append(input_c)
            flatten_layers.append(flatten_c)

        input_num = Input(shape=(len(self._numerical_vars),), dtype='float32')
        flatten_layers.append(input_num)
        inputs.append(input_num)

        flatten = concatenate(flatten_layers, axis=-1)

        fc1 = Dense(X.shape[1] * 10, kernel_initializer='normal')(flatten)
        fc1 = Activation('relu')(fc1)
        # fc1 = BatchNormalization(fc1)
        # fc1 = Dropout(0.75)(fc1)

        fc2 = Dense(X.shape[1] * 5, kernel_initializer='normal')(fc1)
        fc2 = Activation('relu')(fc2)
        # fc2 = BatchNormalization(fc2)
        # fc2 = Dropout(0.5)(fc2)

        output = Dense(1, activation='sigmoid')(fc2)
        nnet = Model(inputs=inputs, outputs=output)

        return nnet
