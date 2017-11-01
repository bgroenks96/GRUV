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

parser = argparse.ArgumentParser(description="Generate song from current saved training data.")
parser.add_argument("--batch", default=1, type=int, help="Number of generations to run.")
parser.add_argument("--iteration", default=0, type=int, help="Current training iteration load weights for.")
parser.add_argument("--seqlen", default=10, type=int, help="Sequence length.")
parser.add_argument("--use-train", action='store_true', help='True if training data should be sampled to seed generation. Defaults to false (use generation data).')
args = parser.parse_args()

config = nn_config.get_neural_net_configuration()
sample_frequency = config['sampling_frequency']
if args.use_train:
    inputFile = config['model_file']
else:
    inputFile = config['gen_file']
model_basename = config['model_basename']
cur_iter = args.iteration
gen_count = args.batch
model_filename = model_basename + str(cur_iter)
output_filename = './generated_song'
output_file_ext = '.wav'

#Load up the training data
if args.use_train:
    print ('Loading training data')
else:
    print ('Loading generation data')
#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
#X_var is a matrix of size (num_frequency_dims,) containing the variance for each frequency dimension
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
X_mean = np.load(inputFile + '_mean.npy')
X_var = np.load(inputFile + '_var.npy')
print ('Finished loading data')

#Figure out how many frequencies we have in the data
num_timesteps = X_train.shape[1]
freq_space_dims = X_train.shape[2]
hidden_dims = config['hidden_dimension_size']

#Creates a lstm network
print('Initializing network...')
model = network_utils.create_lstm_network(num_timesteps=num_timesteps, num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims)
#You could also substitute this with a RNN or GRU
#model = network_utils.create_gru_network()

#Load existing weights if available
if os.path.isfile(model_filename):
    print('Loading weights from file {0}'.format(model_filename))
    model.load_weights(model_filename)
else:
	print('Model filename ' + model_filename + ' could not be found!')

max_seq_len = args.seqlen; #Defines how long the final song is. Total song length in samples = max_seq_len * example_len
print ('Starting generation!')
#Here's the interesting part
#We need to create some seed sequence for the algorithm to start with
#Currently, we just grab an existing seed sequence from our training data and use that
#However, this will generally produce verbatum copies of the original songs
#In a sense, choosing good seed sequences = how you get interesting compositions
#There are many, many ways we can pick these seed sequences such as taking linear combinations of certain songs
#We could even provide a uniformly random sequence, but that is highly unlikely to produce good results
for i in range(gen_count):
    seed_len = 2
    seed_seq = seed_generator.generate_copy_seed_sequence(seed_length=seed_len, training_data=X_train)
    output = sequence_generator.generate_from_seed(model=model, seed=seed_seq, sequence_length=max_seq_len, data_variance=X_var, data_mean=X_mean)
    #Save the generated sequence to a WAV file
    save_generated_example('{0}_{1}{2}'.format(output_filename, i, output_file_ext), output, sample_frequency=sample_frequency)
    
print ('Finished generation!')
