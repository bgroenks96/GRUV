from __future__ import absolute_import
from __future__ import print_function
from generate import generate_from_seeds
from keras.callbacks import EarlyStopping
import numpy as np
import os
import nn_utils.network_utils as network_utils
import config.nn_config as nn_config
import gen_utils.seed_generator as seed_generator
import gen_utils.sequence_generator as sequence_generator
import tensorflow as tf
import argparse

config = nn_config.get_autoencoder_gan_configuration()

parser = argparse.ArgumentParser(description="Train the NuGRUV GAN against the current dataset.")
parser.add_argument("dataset_name", help="Name of the dataset to use for training.")
parser.add_argument("-s", "--start-iter", default=0, type=int, help="Current training iteration to start from.")
parser.add_argument("-n", "--num-iters", default=10, type=int, help='Number of training iterations to run.')
parser.add_argument("--dis-epochs", default=10, type=int, help="Number of epochs per iteration to train the discriminator.")
parser.add_argument("--enc-epochs", default=20, type=int, help="Number of epochs per iteration of the autoencoder.")
parser.add_argument("--com-epochs", default=1, type=int, help="Number of epochs per iteration to train the combined GAN model.")
parser.add_argument("--dis-samples", default=10, type=int, help="Number of samples to generate for the discriminator to train against on each iteration.")
parser.add_argument("-b", "--batch", default=64, type=int, help="Number of seeds to generate per iteration.")
parser.add_argument("--skip-validation", action="store_true", help="Do not use cross validation data.")
parser.add_argument("-i", "--interval", default=10, type=int, help="Number of iterations to run in between retaining saved weights.")
parser.add_argument("-r", "--run", default=0, type=int, help="Integer id for this run (used for weight files). Defaults to zero.")
args = parser.parse_args()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

input_file = config['dataset_directory'] + args.dataset_name + '/' + args.dataset_name
cur_iter = args.start_iter
dis_basename = config['model_basename'] + str(args.run) + '-Discriminator_'
dis_filename = dis_basename + str(cur_iter)
gen_basename = config['model_basename'] + str(args.run) + '-Generator_'
gen_filename = gen_basename + str(cur_iter)
enc_basename = config['model_basename'] + str(args.run) + '-Encoder_'
enc_filename = enc_basename + str(cur_iter)
dec_basename = config['model_basename'] + str(args.run) + '-Decoder_'
dec_filename = dec_basename + str(cur_iter)
skip_validation = args.skip_validation

# Load up the training data
print('Loading training data')
# X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
# y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
X_train = np.load(input_file + '_x.npy')
y_train = X_train
if not skip_validation:
    X_val = np.load(input_file + '_val_x.npy')
    y_val = X_val
print('Finished loading training data')

# Figure out how many frequencies we have in the data
num_timesteps = X_train.shape[1]
freq_space_dims = X_train.shape[2]
seed_size = config['generator_seed_size']

# Creates an autoencoder and GAN using the decoder from the autoencoder network
print('Initializing network...')
encoder = network_utils.create_aegan_encoder_network(freq_space_dims, config, num_timesteps)
decoder = network_utils.create_aegan_decoder_network(freq_space_dims, config, num_timesteps)
autoencoder, gan = network_utils.create_autoencoder_gan(seed_size, encoder, decoder, freq_space_dims, config, num_timesteps)

print('Model summary:')
print('==== Autoencoder ====')
autoencoder.summary()
gan.summary()

# Load existing weights if available
if os.path.isfile(dis_filename):
    print('Loading existing weight data (Discriminator) from {}'.format(dis_filename))
    gan.discriminator.load_weights(dis_filename)
if os.path.isfile(gen_filename):
    print('Loading existing weight data (Generator) from {}'.format(gen_filename))
    gan.generator.load_weights(gen_filename)
if os.path.isfile(enc_filename):
    print('Loading existing weight data (Encoder) from {}'.format(enc_filename))
    encoder.load_weights(enc_filename)
if os.path.isfile(dec_filename):
    print('Loading existing weight data (Decoder) from {}'.format(dec_filename))
    decoder.load_weights(dec_filename)

print('Training set shape: {0}'.format(X_train.shape))
val_data = None
if not skip_validation:
    print('Validation set shape: {0}'.format(X_val.shape))
    val_data = (X_val, y_val)

def generate_random_seeds(seed_dims, repeat_count=0, mean=0, std=1):
    seeds = np.random.uniform(low=-1, high=1, size=seed_dims) #np.random.normal(loc=mean, scale=std, size=seed_dims)
    #copies = seeds.copy()
    #for i in xrange(repeat_count - 1):
    #    copies = np.concatenate((copies, seeds), axis=0)
    return seeds

def random_training_examples(X_train, X_val=[], seed_len=1, count=1):
    X_val = np.array(X_val)
    num_train_examples = X_train.shape[0]
    num_val_examples = X_val.shape[0]
    val_split = float(num_val_examples) / (num_train_examples + num_val_examples)
    val_count = int(val_split*count);
    train_count = count - val_count;
    train_examples = np.zeros((0, X_train.shape[1]*seed_len,X_train.shape[2]))
    for i in xrange(train_count):
        next_example = seed_generator.generate_copy_seed_sequence(seed_length=seed_len, training_data=X_train)
        train_examples = np.concatenate((train_examples, next_example), axis=0)
    val_examples = np.zeros((0, X_val.shape[1]*seed_len,X_val.shape[2]))
    for i in xrange(val_count):
        next_example = seed_generator.generate_copy_seed_sequence(seed_length=seed_len, training_data=X_val)
        val_examples = np.concatenate((val_examples, next_example), axis=0)
    return (train_examples, val_examples)

def train_discriminator(X_train, X_val, batch_size, callbacks=[]):
    print('Training discriminator...')
    train_seeds = generate_random_seeds(seed_dims=(X_train.shape[0], seed_size))
    val_seeds = generate_random_seeds(seed_dims=(X_val.shape[0], seed_size))
    X_train_real, X_val_real = random_training_examples(X_train, X_val, seed_len=1)
    X_train_fake = generate_from_seeds(gan.generator, train_seeds, max_seq_len=num_timesteps, uncenter_data=False)
    X_val_fake = generate_from_seeds(gan.generator, val_seeds, max_seq_len=num_timesteps, uncenter_data=False)
    return gan.fit_discriminator(X_train_real, X_train_fake, batch_size=batch_size, epochs=args.dis_epochs, shuffle=False, verbose=1, callbacks=callbacks, validation_data=(X_val_real, X_val_fake))

def print_hist_stats(h):
    loss = h.history['loss']
    print('loss min: {0}  loss max: {1}'.format(min(loss), max(loss)))
    if 'acc' in h.history:
        acc = h.history['acc']
        print('acc min: {0}  acc max: {1}'.format(min(acc), max(acc)))

early_stop = EarlyStopping(monitor='acc', min_delta=0.01, patience=1, verbose=1, mode='max')

print('Starting training...')
# If we're not starting at zero, then bump current iteration up one, assuming we've loaded weights for the starting iteration
if cur_iter > 0:
    cur_iter += 1
num_iters = cur_iter + args.num_iters
hist = {}
while cur_iter < num_iters:
    # Start training iteration for each model
    print('Iteration: {0}'.format(cur_iter))
    train_seeds = generate_random_seeds(seed_dims=(args.batch, seed_size))
    val_seeds = None
    if not args.skip_validation:
        val_seeds = generate_random_seeds(seed_dims=(args.batch, seed_size))
    print('Training autoencoder for {0} epochs (batch size: {1})'.format(args.enc_epochs, 'default'))
    enc_hist = autoencoder.fit(X_train, y_train, epochs=args.enc_epochs, shuffle=True, verbose=1, validation_data=val_data)
    print_hist_stats(enc_hist)
    print('Saving encoder weights for iteration {0} ...'.format(cur_iter))
    encoder.save_weights(enc_basename + str(cur_iter))
    print('Saving decoder weights for iteration {0} ...'.format(cur_iter))
    decoder.save_weights(dec_basename + str(cur_iter))
    print('Training discriminator for {0} epochs with {1} training examples'.format(args.dis_epochs, args.batch))
    dis_hist = train_discriminator(X_train, X_val, args.batch, callbacks=[early_stop])
    print_hist_stats(dis_hist)
    print('Saving discriminator weights for iteration {0} ...'.format(cur_iter))
    gan.discriminator.save_weights(dis_basename + str(cur_iter))
    print('Training generator (combined model) for {0} epochs'.format(args.com_epochs))
    gan_hist = gan.fit(train_seeds, epochs=args.com_epochs, shuffle=True, verbose=1, callbacks=[early_stop], validation_x=val_seeds)
    print_hist_stats(gan_hist)
    print('Saving generator weights (post-train) for iteration {0} ...'.format(cur_iter))
    gan.generator.save_weights(gen_basename + str(cur_iter))

    # hist[cur_iter] = {'gen' : gen_hist.history, 'dec' : dec_hist.history, 'com' : gan_hist.history}
    # np.save('metrics-train-aegan-{0}.npy'.format(args.run), hist)

    # Clean weights from last iteration, if between persistent save intervals
    last_iter = cur_iter - 1
    if last_iter >= 0 and last_iter % args.interval != 0:
        if os.path.isfile(gen_basename + str(last_iter)):
            os.remove(gen_basename + str(last_iter))
        if os.path.isfile(dis_basename + str(last_iter)):
            os.remove(dis_basename + str(last_iter))
        if os.path.isfile(enc_basename + str(last_iter)):
            os.remove(enc_basename + str(last_iter))
        if os.path.isfile(dec_basename + str(last_iter)):
            os.remove(dec_basename + str(last_iter))

    cur_iter += 1

print('Training complete!')

# Deprecated, dynamic validation splitting. Incurs a lot of additional computational overhead per training recurrent_units
# ...
#def prepare_validation_split(x_train, y_train, x_val, y_val, validate_size):
    #if validate_size == 0:
        #return x_train, y_train, x_val, y_val
    #sample_max = x_train.shape[0] - validate_size
    #stride = x_train.shape[0] / validate_size
    #for i in xrange(0, sample_max, stride):
        #x_val = np.concatenate((x_val, np.reshape(x_train[i], (1, x_train.shape[1], x_train.shape[2]))), axis=0)
        #y_val = np.concatenate((y_val, np.reshape(y_train[i], (1, y_train.shape[1], y_train.shape[2]))), axis=0)
        #x_train = np.delete(x_train, i, axis=0)
        #y_train = np.delete(y_train, i, axis=0)
    #return x_train, y_train, x_val, y_val

## Set up validation split
#validate_size = int(round(X_train.shape[0] * validation_split))
#assert X_train.shape[0] == y_train.shape[0]
#X_val = np.zeros((0, X_train.shape[1], X_train.shape[2]))
#y_val = np.zeros((0, y_train.shape[1], y_train.shape[2]))
#print('Splitting data for validation...')
#data_parts = prepare_validation_split(X_train, y_train, X_val, y_val, validate_size)
#X_train = data_parts[0]
#y_train = data_parts[1]
#X_val = data_parts[2]
#y_val = data_parts[3]
