from keras.models import Sequential, Model
from keras.layers import Input, Conv1D, LSTM, TimeDistributed, Dense
from keras.layers import GaussianDropout, Lambda, GRU, Reshape, PReLU
import numpy as np

def create_aegan_encoder_network(num_frequency_dimensions, config, num_timesteps=None, batch_size=None, stateful=False):
    num_hidden_dimensions = config['encoding_hidden_dims']
    dropout = GaussianDropout(config['autoencoder_dropout'])
    inputs = Input(batch_shape=(batch_size, num_timesteps, num_frequency_dimensions))
    conv_in = Conv1D(num_hidden_dimensions, kernel_size=2, activation='tanh', padding='causal')(inputs)
    td_dense = TimeDistributed(Dense(num_hidden_dimensions, activation='tanh'))(dropout(conv_in))
    lstm = LSTM(num_hidden_dimensions, return_sequences=True, activation='tanh', stateful=stateful)(dropout(td_dense))
    model = Model(inputs=inputs, outputs=lstm)
    return model

def create_aegan_decoder_network(num_frequency_dimensions, config, num_timesteps=None, batch_size=None, stateful=False):
    num_hidden_dimensions = config['encoding_hidden_dims']
    dropout = GaussianDropout(config['autoencoder_dropout'])
    inputs = Input(batch_shape=(batch_size, num_timesteps, num_hidden_dimensions))
    #dense_in = Dense(num_hidden_dimensions, activation='tanh')(inputs)
    #reshaped = Reshape((num_timesteps, num_hidden_dimensions))(dense_in)
    td_dense = TimeDistributed(Dense(num_hidden_dimensions, activation='tanh'))(dropout(inputs))
    conv_out = Conv1D(num_frequency_dimensions, kernel_size=2, padding='causal')(dropout(td_dense))
    model = Model(inputs=inputs, outputs=conv_out)
    return model

def create_aegan_generator_network(seed_size, decoder, config, num_timesteps, batch_size=None, stateful=False):
    num_hidden_dimensions = config['encoding_hidden_dims']
    dropout = GaussianDropout(config['generator_dropout'])
    inputs = Input(batch_shape=(batch_size, seed_size))
    dense_in = Dense(num_hidden_dimensions*num_timesteps, activation='tanh')(inputs)
    reshaped = Reshape((num_timesteps, num_hidden_dimensions))(dropout(dense_in))
    dense_h1 = TimeDistributed(Dense(num_hidden_dimensions, activation='tanh'))(dropout(reshaped))
    dense_out = TimeDistributed(Dense(num_hidden_dimensions, activation='tanh'))(dense_h1)
    gen_model = Model(inputs=inputs, outputs=dense_out)
    aegan_model = Sequential()
    aegan_model.add(gen_model)
    aegan_model.add(decoder)
    return aegan_model

# Prototype gneerator network that deconvolves a frequency domain signal with 'num_frequency_dimensions' and 'num_timesteps'
# from a random seed of size 'seed_size'.
def create_deconvolutional_generator_network(seed_size, batch_size, num_frequency_dimensions, num_timesteps, config, stateful=True):
    num_hidden_dimensions = config['generator_hidden_dims']
    inputs = Input(batch_shape=(batch_size, seed_size))
    dense_in = Dense(num_hidden_dimensions*num_timesteps, activation='relu')(inputs)
    conv_h1 = Conv1D(num_hidden_dimensions, kernel_size=2, activation='relu', padding='causal')(Reshape((num_timesteps, num_hidden_dimensions))(dense_in))
    lstm = LSTM(num_hidden_dimensions, return_sequences=True, activation='relu', stateful=stateful)(conv_h1)
    conv_out = Conv1D(num_frequency_dimensions, kernel_size=2, padding='causal')(lstm)
    model = Model(inputs=inputs, outputs=conv_out)
    model.compile(loss='logcosh', optimizer=config['generator_optimizer'])
    return model

# GAN friendly adaptation of Nayebi et Vitelli's regression concept; replaces TDD input/output layers with
# 1D convolutional layers and uses two hidden LSTM layers.
def create_regression_generator_network(num_frequency_dimensions, config, num_timesteps=None, batch_size=None, stateful=False):
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

# Simple discriminator that applies TDD and then flattens the time dimensions before computing the probability
# of the sample being real or fake.
def create_discriminator_network(num_frequency_dimensions, num_timesteps, config, batch_size=None):
    num_hidden_dimensions = config['discriminator_hidden_dims']
    dropout = GaussianDropout(['discriminator_dropout'])
    inputs = Input(batch_shape=(batch_size, num_timesteps, num_frequency_dimensions))
    conv_in = Conv1D(num_hidden_dimensions, kernel_size=2, activation='relu', padding='causal')(inputs)
    conv_h1 = Conv1D(num_hidden_dimensions, kernel_size=2, activation='relu', padding='causal')(dropout(conv_in))
    td_dense = TimeDistributed(Dense(num_hidden_dimensions, activation='relu'))(dropout(conv_h1))
    lstm = LSTM(1, activation='sigmoid', return_sequences=False)(dropout(td_dense))
    #dense_out = Dense(1, activation='sigmoid')(dropout(lstm))
    discriminator = Model(inputs=inputs, outputs=lstm)
    discriminator.compile(loss='binary_crossentropy', optimizer=config['discriminator_optimizer'], metrics=['accuracy'])
    return discriminator

def create_autoencoder_gan(seed_size, encoder, decoder, num_frequency_dimensions, config, num_timesteps, batch_size=None, stateful=False):
    # Create full autoencoder network
    autoencoder = Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)
    # Create generator network
    generator = create_aegan_generator_network(seed_size, decoder, config, num_timesteps=num_timesteps, batch_size=batch_size, stateful=stateful)
    # Create discriminator network
    discriminator = create_discriminator_network(num_frequency_dimensions, num_timesteps, config)
    # Create GAN (combined model)
    decoder.trainable = False
    gan = GAN(generator, discriminator, config)
    decoder.trainable = True
    autoencoder.compile(loss='mean_squared_error', optimizer=config['autoencoder_optimizer'])
    return (autoencoder, gan)

# Constructs a GAN using the regression generator.
def create_regression_gan(num_frequency_dimensions, config, num_timesteps=None, batch_size=None, stateful=False):
    # Create generator network
    generator = create_regression_generator_network(num_frequency_dimensions, config, num_timesteps=num_timesteps, batch_size=batch_size, stateful=stateful)
    # Create discriminator network
    discriminator = create_discriminator_network(num_frequency_dimensions, num_timesteps, config)
    # Create GAN (combined model)
    return GAN(generator, discriminator, config)

# Constructs a GAN using the deconvolutional (seed -> output) generator.
def create_deconvolutional_gan(seed_size, batch_size, num_frequency_dimensions, num_timesteps, config, stateful=True):
    # Create generator network
    generator = create_deconvolutional_generator_network(seed_size, batch_size, num_frequency_dimensions, num_timesteps, config, stateful)
    # Create discriminator (or "discriminator") network
    discriminator = create_discriminator_network(num_frequency_dimensions, num_timesteps, config, batch_size=batch_size)
    # Create GAN (combined model)
    return GAN(generator, discriminator, config)

# Original single layer, LSTM autoencoder network proposed by Nayebi et Vitelli (2015)
def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(LSTM(units=num_hidden_dimensions, return_sequences=True))
    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

# Original single layer, GRU autoencoder network proposed by Nayebi et Vitelli (2015)
def create_gru_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(GRU(units=num_hidden_dimensions, return_sequences=True))
    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

# Simple time-distributed network that just adds Gaussian noise to the input signal.
def create_noise_network(num_frequency_dimensions):
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: np.random.choice((0,1), p=[0.1,0.9])*np.random.normal(1,0.3)*x*0.5), input_shape=(None, num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

# Simple implementation of a Generative Adverserial Network taking an arbitrary generator and discriminator network.
class GAN:
    def __init__(self, generator, discriminator, config):
        model = Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=config['combined_optimizer'], metrics=['accuracy'])
        discriminator.trainable = True
        self.generator = generator
        self.discriminator = discriminator
        self.model = model

    # Fit the generator network against the given training data
    def fit_generator(self, X_train, y_train, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_data=None):
        return self.generator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=validation_data)

    # Fit the discriminator network against the given real and fake X examples. An example output of '1' will be generated for each real example, and '0' for each fake one.
    def fit_discriminator(self, X_real, X_fake, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_data=[]):
        num_real = X_real.shape[0]
        num_fake = X_fake.shape[0]
        X_train = np.concatenate((X_real, X_fake), axis=0)
        y_train = np.concatenate((np.ones(num_real), np.zeros(num_fake)))
        val_data = None
        if len(validation_data) > 0:
            val_real = validation_data[0]
            val_fake = validation_data[1]
            X_val = np.concatenate((val_real, val_fake))
            y_val = np.concatenate((np.ones(val_real.shape[0]), np.zeros(val_fake.shape[0])))
            val_data = (X_val, y_val)
        return self.discriminator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=val_data)

    def fit(self, X_train, batch_size=None, epochs=10, shuffle=False, verbose=1, callbacks=None, validation_x=[]):
        num_examples = X_train.shape[0]
        num_timesteps = X_train.shape[1]
        y_train = np.ones(num_examples)
        val_data = None
        if len(validation_x) > 0:
            num_val = validation_x.shape[0]
            y_val = np.ones(num_val)
            val_data = (validation_x, y_val)
        self.discriminator.trainable = False
        hist = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, callbacks=callbacks, validation_data=val_data)
        self.discriminator.trainable = True
        return hist

    def summary(self):
        print('==== Generator ====')
        self.generator.summary()
        print('==== Discriminator ====')
        self.discriminator.summary()
        print('==== Combined (GAN) ====')
        self.discriminator.trainable = False
        self.model.summary()
        self.discriminator.trainable = True
