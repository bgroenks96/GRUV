from keras.models import *
from keras.layers import *
from keras import optimizers
import numpy as np

def create_lstm_network(num_frequency_dimensions, config):
    inputs = Input(shape=(None, num_frequency_dimensions))
    #td_input = TimeDistributed(Dense(num_hidden_dimensions))(inputs)
    
    ## LSTM upper layer
    #lstm_1_1 = LSTM(num_hidden_dimensions, return_sequences=True)(GaussianDropout(dropout_rate)(td_input))
    #lstm_1_2 = LSTM(num_hidden_dimensions / 2, return_sequences=True)(GaussianNoise(0.2)(lstm_1_1))
    ## LSTM lower layer
    #lstm_2_1 = LSTM(num_hidden_dimensions, return_sequences=True)(GaussianDropout(dropout_rate)(td_input))
    #lstm_2_2 = LSTM(num_hidden_dimensions / 2, return_sequences=True)(GaussianNoise(0.2)(lstm_2_1))
    ## Merge mult
    #add = Add()([lstm_1_2, lstm_2_2])
    
    num_hidden_dimensions = config['generator_hidden_dims']
    dropout_rate = config['generator_dropout']
    conv_in = Conv1D(num_hidden_dimensions, kernel_size=2, activation='tanh', padding='causal')(inputs)
    lstm_1 = LSTM(num_hidden_dimensions, return_sequences=True)(GaussianDropout(dropout_rate)(conv_in))
    lstm_2 = LSTM(num_hidden_dimensions, return_sequences=True)(GaussianDropout(dropout_rate)(lstm_1))
    conv_out = Conv1D(num_frequency_dimensions, kernel_size=2, activation='tanh', padding='causal')(lstm_2)
    
    # Convert back to frequency space
    #td_output = TimeDistributed(Dense(num_frequency_dimensions))(lstm_2)
    model = Model(inputs=inputs, outputs=conv_out)
    model.compile(loss='logcosh', optimizer=config['generator_optimizer'])
    return model

def create_gru_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1, optimizer='rmsprop', dropout_rate=0.3):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    model.add(GaussianDropout(dropout_rate))
    for cur_unit in xrange(num_recurrent_units):
        model.add(GRU(units=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
    
def create_noise_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: np.random.choice((0,1), p=[0.1,0.9])*np.random.normal(1,0.3)*x*0.5), input_shape=(None, num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def create_gan(num_frequency_dimensions, config):
    # Create generator network
    generator = create_lstm_network(num_frequency_dimensions, config)
    # Create decoder (or "discriminator") network
    num_hidden_dimensions = config['decoder_hidden_dims']
    inputs = Input(shape=(None, num_frequency_dimensions))
    conv_in = Conv1D(num_hidden_dimensions, kernel_size=2, activation='tanh', padding='same')(inputs)
    avg_pool_1 = AveragePooling1D(pool_size=2)(conv_in)
    conv_hidden_1 = Conv1D(num_hidden_dimensions / 2, kernel_size=2, activation='tanh', padding='same')(avg_pool_1)
    avg_pool_2 = AveragePooling1D(pool_size=2)(conv_hidden_1)
    conv_hidden_2 = Conv1D(num_hidden_dimensions / 4, kernel_size=2, activation='tanh', padding='same')(avg_pool_2)
    lstm = LSTM(num_hidden_dimensions)(conv_hidden_2)
    dense_out = Dense(1, activation='sigmoid')(lstm)
    decoder = Model(inputs=inputs, outputs=dense_out)
    decoder.compile(loss='binary_crossentropy', optimizer=config['decoder_optimizer'], metrics=['accuracy'])
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
    def fit_generator(self, X_train, y_train, batch_size=None, epochs=10, shuffle=False, verbose=1, validation_data=None):
        return self.generator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_data=validation_data)
    
    # Fit the decoder network against the given real and fake X examples. An example output of '1' will be generated for each real example, and '0' for each fake one.
    def fit_decoder(self, X_real, X_fake, batch_size=None, epochs=10, shuffle=False, verbose=1, validation_split=0.0):
        num_real = X_real.shape[0]
        num_fake = X_fake.shape[0]
        X_train = np.concatenate((X_real, X_fake), axis=0)
        y_train = np.concatenate((np.ones((num_real, 1)), np.zeros((num_fake, 1))), axis=0)
        return self.decoder.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_split=validation_split)
    
    def fit(self, X_train, batch_size=None, epochs=10, shuffle=False, verbose=1, validation_split=0.0):
        num_examples = X_train.shape[0]
        num_timesteps = X_train.shape[1]
        y_train = np.ones((num_examples, 1))
        self.decoder.trinable = False
        hist = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_split=validation_split)
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
        
