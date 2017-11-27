from keras.models import *
from keras.layers import *
from keras import initializers
from keras import optimizers
import numpy as np

def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(GaussianDropout(dropout_rate))
    model.add(LSTM(units=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def create_gru_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(GRU(units=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def create_noise_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: np.random.choice((0,1), p=[0.1,0.9])*np.random.normal(1,0.3)*x*0.5), input_shape=(None, num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def create_autoencoding_generator_network(num_frequency_dimensions, num_timesteps, config, batch_size=None, stateful=False):
    inputs = Input(batch_shape=(batch_size, num_timesteps, num_frequency_dimensions))
    num_hidden_dimensions = config['generator_hidden_dims']
    dropout = GaussianDropout(config['generator_dropout'])
    conv_in = Conv1D(num_hidden_dimensions, kernel_size=2, padding='causal')(inputs)
    lstm_1 = LSTM(num_hidden_dimensions, return_sequences=True, stateful=stateful)(dropout(conv_in))
    lstm_2 = LSTM(num_hidden_dimensions, return_sequences=True, stateful=stateful)(dropout(lstm_1))
    td_dense = TimeDistributed(Dense(num_hidden_dimensions))(lstm_2)
    conv_out = Conv1D(num_frequency_dimensions, kernel_size=2, padding='causal')(dropout(td_dense))
    model = Model(inputs=inputs, outputs=conv_out)
    model.compile(loss='mean_squared_error', optimizer=config['generator_optimizer'])
    return model

def create_decoder_network(num_frequency_dimensions, num_timesteps, config):
    num_hidden_dimensions = config['decoder_hidden_dims']
    assert num_hidden_dimensions % 2 == 0
    dropout = GaussianDropout(['decoder_dropout'])
    inputs = Input(shape=(num_timesteps, num_frequency_dimensions))
    conv_in = Conv1D(num_hidden_dimensions, kernel_size=2, activation='tanh', padding='causal')(inputs)
    dense_h0 = TimeDistributed(Dense(num_hidden_dimensions, activation='tanh'))(conv_in)
    dense_h1 = TimeDistributed(Dense(num_hidden_dimensions, activation='tanh'))(conv_in)
    lstm = LSTM(num_hidden_dimensions, return_sequences=False, activation='tanh')(dropout(dense_h0))
    dense_out = Dense(1, activation='sigmoid')(dropout(lstm))
    decoder = Model(inputs=inputs, outputs=dense_out)
    decoder.compile(loss='binary_crossentropy', optimizer=config['decoder_optimizer'], metrics=['accuracy'])
    return decoder

def create_autoencoder_gan(num_frequency_dimensions, num_timesteps, config, batch_size=None, stateful=False):
    # Create generator network
    generator = create_autoencoding_generator_network(num_frequency_dimensions, num_timesteps, config, batch_size, stateful)
    # Create decoder (or "discriminator") network
    decoder = create_decoder_network(num_frequency_dimensions, num_timesteps, config)
    # Create GAN (combined model)
    return GAN(generator, decoder, config)

class GAN:
    def __init__(self, generator, decoder, config):
        model = Sequential()
        model.add(generator)
        decoder.trainable = False
        model.add(decoder)
        model.compile(loss='binary_crossentropy', optimizer=config['combined_optimizer'], metrics=['accuracy'])
        decoder.trainable = True
        self.generator = generator
        self.decoder = decoder
        self.model = model

    # Fit the generator network against the given training data
    def fit_generator(self, X_train, y_train, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_data=None):
        return self.generator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=validation_data)

    # Fit the decoder network against the given real and fake X examples. An example output of '1' will be generated for each real example, and '0' for each fake one.
    def fit_decoder(self, X_real, X_fake, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_data=[]):
        num_real = X_real.shape[0]
        num_fake = X_fake.shape[0]
        X_train = np.concatenate((X_real, X_fake), axis=0)
        y_train = np.concatenate((np.ones((num_real, 1)), np.zeros((num_fake, 1))), axis=0)
        val_data = None
        if len(validation_data) > 0:
            val_real = validation_data[0]
            val_fake = validation_data[1]
            X_val = np.concatenate((val_real, val_fake), axis=0)
            y_val = np.concatenate((np.ones((val_real.shape[0], 1)), np.zeros((val_fake.shape[0], 1))), axis=0)
            val_data = (X_val, y_val)
        return self.decoder.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=val_data)

    def fit(self, X_train, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_x=[]):
        num_examples = X_train.shape[0]
        num_timesteps = X_train.shape[1]
        y_train = np.ones((num_examples, 1))
        val_data = None
        if len(validation_x) > 0:
            num_val = validation_x.shape[0]
            y_val = np.ones((num_val, 1))
            val_data = (validation_x, y_val)
        self.decoder.trinable = False
        hist = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=val_data)
        self.decoder.trainable = True
        return hist

    def summary(self):
        print('==== Generator ====')
        self.generator.summary()
        print('==== Decoder ====')
        self.decoder.summary()
        print('==== Combined (GAN) ====')
        self.decoder.trainable = False
        self.model.summary()
        self.decoder.trainable = True
