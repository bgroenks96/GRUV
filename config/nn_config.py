from keras import optimizers

def get_default_configuration():
    nn_params = {}
    nn_params['sampling_frequency'] = 44100
    nn_params['generator_hidden_dims'] = 256
    nn_params['generator_dropout'] = 0.2
    nn_params['generator_optimizer'] = 'adam'
    #The weights filename for saving/loading trained models
    nn_params['model_basename'] = './NuGruvModelWeights'
    #The training data directory
    nn_params['dataset_directory'] = './datasets/'
    return nn_params

def get_autoencoder_gan_configuration():
    nn_params = {}
    nn_params['sampling_frequency'] = 44100
    nn_params['autoencoder_optimizer'] = 'adam'
    nn_params['autoencoder_dropout'] = 0.2
    nn_params['encoding_hidden_dims'] = 256
    nn_params['generator_seed_size'] = 256
    nn_params['generator_dropout'] = 0.1
    nn_params['generator_optimizer'] = 'adam'
    nn_params['discriminator_dropout'] = 0.3
    nn_params['discriminator_optimizer'] = 'adam'
    nn_params['discriminator_hidden_dims'] = 64
    nn_params['combined_optimizer'] = 'adam'
    #The weights filename for saving/loading trained models
    nn_params['model_basename'] = './NuGruvModelWeights'
    #The training data directory
    nn_params['dataset_directory'] = './datasets/'
    return nn_params

def get_regression_gan_configuration():
    nn_params = {}
    nn_params['sampling_frequency'] = 44100
    nn_params['generator_hidden_dims'] = 256
    nn_params['generator_dropout'] = 0.2
    nn_params['generator_optimizer'] = 'adam'
    nn_params['discriminator_dropout'] = 0.3
    nn_params['discriminator_optimizer'] = 'adam'
    nn_params['discriminator_hidden_dims'] = 64
    nn_params['combined_optimizer'] = 'adam'
    #The weights filename for saving/loading trained models
    nn_params['model_basename'] = './NuGruvModelWeights'
    #The training data directory
    nn_params['dataset_directory'] = './datasets/'
    return nn_params

def get_deconv_gan_configuration():
    nn_params = {}
    nn_params['sampling_frequency'] = 44100
    nn_params['generator_hidden_dims'] = 1024
    nn_params['generator_dropout'] = 0.1
    nn_params['generator_optimizer'] = 'adam'
    nn_params['discriminator_dropout'] = 0.4
    nn_params['discriminator_optimizer'] = optimizers.SGD(0.01, momentum=0.1)
    nn_params['discriminator_hidden_dims'] = 128
    nn_params['combined_optimizer'] = optimizers.Adadelta(0.05)
    #The weights filename for saving/loading trained models
    nn_params['model_basename'] = './NuGruvModelWeights'
    #The training data directory
    nn_params['dataset_directory'] = './datasets/'
    return nn_params
