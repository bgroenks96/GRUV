from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils
import gen_utils.seed_generator as seed_generator
import gen_utils.sequence_generator as sequence_generator
from data_utils.parse_files import *
import config.nn_config as nn_config
import argparse

# Generate a new sequence using the given model, seed dataset, and generation parameters.
def generate_from_data(model, x_data, max_seq_len, seed_len=1, gen_count=1, include_raw_seed=False, include_model_seed=False, uncenter_data=False, X_var=None, X_mean=None):
    print('Starting generation!')
    #Here's the interesting part
    #We need to create some seed sequence for the algorithm to start with
    #Currently, we just grab an existing seed sequence from our training data and use that
    #However, this will generally produce verbatum copies of the original songs
    #In a sense, choosing good seed sequences = how you get interesting compositions
    #There are many, many ways we can pick these seed sequences such as taking linear combinations of certain songs
    #We could even provide a uniformly random sequence, but that is highly unlikely to produce good results
    outputs = []
    for i in xrange(gen_count):
        print("Generating sample {0}/{1}".format(i+1, gen_count))
        seed_seq = seed_generator.generate_copy_seed_sequence(seed_length=seed_len, training_data=x_data)
        output = sequence_generator.generate_from_example_seed(model, seed_seq, max_seq_len, include_raw_seed, include_model_seed, uncenter_data, X_var, X_mean)
        outputs.append(output)
    model.reset_states() # If model is stateful, states should be reset
    print('Finished generation!')
    return np.array(outputs)
    
def generate_from_seeds(model, x_seeds, max_seq_len, batch_size=None, uncenter_data=False, X_var=None, X_mean=None):
    print('Starting generation!')
    outputs = sequence_generator.generate_from_random_seed(model, x_seeds, max_seq_len, batch_size=batch_size, uncenter_data=uncenter_data, target_mean=X_mean, target_variance=X_var)
    print('Finished generation!')
    return np.array(outputs)
    
def __main__():
    parser = argparse.ArgumentParser(description="Generate song from current saved training data.")
    parser.add_argument("dataset", default='train', type=str, help='The dataset to draw from. Defaults to "train".')
    parser.add_argument("-m", "--model", default='aegan', type=str, help="Model type to use. Defaults to 'aegan' (autoencoder GAN). Can also be dgan (deconvolutional GAN) or 'lstm' for vanilla LSTM model.")
    parser.add_argument("--batch", default=1, type=int, help="Number of generations to run.")
    parser.add_argument("--iteration", default=0, type=int, help="Current training iteration load weights for.")
    parser.add_argument("--seqlen", default=10, type=int, help="Generated sequence length.")
    parser.add_argument("--seedlen", default=1, type=int, help="Length of the seed selected for the generation process.")
    parser.add_argument("--output", default='new', type=str, help="Either 'new' (default) for only new generated output, 'gen' to also include the model's reproduction of the seed, or 'all' to also include the raw seed sequence.")
    parser.add_argument("-r", "--run", default=0, type=int, help="Integer id for this run (used for weight files). Defaults to zero.")
    args = parser.parse_args()
    
    if args.model == 'aegan':
        config = nn_config.get_autoencoder_gan_configuration()
    elif args.model == 'dgan':
        config = nn_config.get_deconv_gan_configuration()
    elif args.model == 'lstm':
        config = nn_config.get_default_configuration()
    else:
        raise(Exception('invalid model type'))

    sample_frequency = config['sampling_frequency']
    input_file = config['dataset_directory'] + args.dataset + '/' + args.dataset
    model_basename = config['model_basename'] + str(args.run) + '_'
    cur_iter = args.iteration
    gen_count = args.batch
    model_filename = model_basename + str(cur_iter)
    output_filename = './generated_song'
    include_model_seed = args.output == 'gen' or args.output == 'all'
    include_raw_seed = args.output == 'all'

    #Load up the training data
    if args.dataset == 'train':
        print('Loading training data')
    else:
        print('Loading generation data')
    #X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
    #y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
    #X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
    #X_var is a matrix of size (num_frequency_dims,) containing the variance for each frequency dimension
    X_train = np.load(input_file + '_x.npy')
    y_train = np.load(input_file + '_y.npy')
    X_mean = np.load(input_file + '_mean.npy')
    X_var = np.load(input_file + '_var.npy')
    print('Finished loading data')
    
    #Figure out how many frequencies we have in the data
    num_timesteps = X_train.shape[1]
    freq_space_dims = X_train.shape[2]

    #Creates a lstm network
    print('Initializing network...')
    model = network_utils.create_deconvolutional_generator_network(256, 1, freq_space_dims, num_timesteps, config, stateful=True)
    #model = network_utils.create_autoencoding_generator_network(num_frequency_dimensions=freq_space_dims, config=config)
    #model = network_utils.create_noise_network(num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims)
    #You could also substitute this with a RNN or GRU
    #model = network_utils.create_gru_network()

    print('Model summary:')
    model.summary()

    #Load existing weights if available
    if os.path.isfile(model_filename):
        print('Loading weights from file {0}'.format(model_filename))
        model.load_weights(model_filename)
    else:
        print('Model filename ' + model_filename + ' could not be found!')

    seq_len = args.seqlen; #Defines how long the final generated song is. Total song length in samples = seq_len * example_len

    x_seeds = np.random.uniform(low=-1, high=1, size=(1,1,256))
    outputs = generate_from_seeds(model, x_seeds, max_seq_len=seq_len, batch_size=1, uncenter_data=True, X_mean=X_mean, X_var=X_var)
    #outputs = generate(model, X_train, seq_len, seed_len=args.seedlen, gen_count=gen_count, include_raw_seed=include_raw_seed, include_model_seed=include_model_seed, uncenter_data=True, X_var=X_var, X_mean=X_mean)
    for i in xrange(gen_count):
        #Save the generated sequence to a WAV file
        save_generated_example('{0}_{1}.wav'.format(output_filename, i), outputs[i], sample_frequency=sample_frequency)
        
if __name__ == '__main__':
    __main__()
