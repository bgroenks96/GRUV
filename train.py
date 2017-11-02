from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils
import config.nn_config as nn_config
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="Train the GRUV network against the current dataset.")
parser.add_argument("current_iteration", default=0, type=int, help="Current training iteration to start from.")
parser.add_argument("num_iterations", default=50, type=int, help="Total number of iterations to perform.")
parser.add_argument("-e", "--epochs", default=25, type=int, help="Number of epochs per iteration.")
parser.add_argument("-b", "--batch", default=1, type=int, help="Training batch size. Larger = more memory, but supposedly trains faster.")
args = parser.parse_args()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

config = nn_config.get_neural_net_configuration()
inputFile = config['model_file']
cur_iter = args.current_iteration
model_basename = config['model_basename']
model_filename = model_basename + str(cur_iter)

#Load up the training data
print ('Loading training data')
#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
#y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
print ('Finished loading training data')

#Figure out how many frequencies we have in the data
num_timesteps = X_train.shape[1]
freq_space_dims = X_train.shape[2]
hidden_dims = config['hidden_dimension_size']
recurrent_units = config['hidden_recurrent_layers']

#Creates a lstm network
print('Initializing network...')
model = network_utils.create_lstm_network(num_timesteps=num_timesteps, num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims, num_recurrent_units=recurrent_units)
#You could also substitute this with a RNN or GRU
#model = network_utils.create_gru_network()

#Load existing weights if available
if os.path.isfile(model_filename):
    print ('Loading existing weight data from {}'.format(model_filename))
    model.load_weights(model_filename)

num_iters = cur_iter + args.num_iterations 		#Number of iterations for training
epochs_per_iter = args.epochs	                #Number of iterations before we save our model
batch_size = args.batch             			#Number of training examples pushed to the GPU per batch.
                                                #Larger batch sizes require more memory, but training will be faster
print ('Starting training!')
while cur_iter < num_iters:
    print('Iteration: ' + str(cur_iter))
    #We set cross-validation to 0,
    #as cross-validation will be on different datasets
    #if we reload our model between runs
    #The moral way to handle this is to manually split
    #your data into two sets and run cross-validation after
    #you've trained the model for some number of epochs
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs_per_iter, verbose=1, validation_split=0.0)
    print(history)
    
    if cur_iter % (epochs_per_iter * 5) != 0 and os.path.isfile(model_basename + str(cur_iter)):
        os.remove(model_basename + str(cur_iter))
        
    cur_iter += epochs_per_iter
    
    print ('Saving weights for iteration {0} ...'.format(cur_iter))
    model.save_weights(model_basename + str(cur_iter))
    
print ('Training complete!')
