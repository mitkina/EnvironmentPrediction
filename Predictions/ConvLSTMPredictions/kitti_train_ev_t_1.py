'''
Code modified from: https://github.com/coxlab/prednet
'''

# Ensure random seed
import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(123)
rn.seed(123)
from keras import backend as K
tf.set_random_seed(123)

import os
from six.moves import cPickle

from time import time

from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hickle as hkl

def get_gradient_norm(model):
    with K.name_scope('gradient_norm'):
        grads = K.gradients(model.total_loss, model.trainable_weights)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
    return norm

# Define loss as MAE of frame predictions after t=0
# It doesn't make sense to compute loss on error representation, since the error isn't wrt ground truth when extrapolating.
def extrap_loss(y_true, y_hat):
    print y_true.shape, y_hat.shape
    y_true = y_true[:, 1:, :, :, 0]
    y_hat = y_hat[:, 1:, :, :, 0]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

save_model = True  # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_full_kitti_weights_ev_t_1.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_full_kitti_model_ev_t_1.json')

# Data files
DATA_DIR_OCC = "../../Data/PredNetData/EvidentialGrids/"

train_file = os.path.join(DATA_DIR_OCC, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR_OCC, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR_OCC, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR_OCC, 'sources_val.hkl')

# Training parameters
nt = 20 # number of timesteps used for sequences in training
nb_epoch = 100
batch_size = 1
samples_per_epoch =  500
N_seq_val = 100  # number of sequences to use for validation

K.set_learning_phase(1) #set learning phase

# Model parameters
n_channels, im_height, im_width = (1, 128, 128)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192) #(n_channels, 12, 24, 48) #24, 48, 96) #(n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0.

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)#, extrap_start_time=10) #pixel_max=3.

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)

val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

print "Shapes: ", train_generator.X.shape, val_generator.X.shape

print "train generator", np.amax(train_generator.X), np.amin(train_generator.X)

lr_schedule = lambda epoch: 0.0001 if epoch < 75 else 0.00001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs

callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

# Append the "l2 norm of gradients" tensor as a metric
# model.metrics_names.append("gradient_norm")
# model.metrics_tensors.append(get_gradient_norm(model))

# tensorboard
tensorboard = TensorBoard(log_dir="logs_check/{}".format(time()), histogram_freq=1, write_graph=False, write_grads=False, write_images=True, batch_size=1)
# np.ceil(val_generator.X.shape[0] / batch_size),
# tensorboard = TensorBoardWrapper(val_generator, 1, batch_size, log_dir="logs/{}".format(time()), histogram_freq=1, batch_size=batch_size, write_graph=False, write_grads=True, write_images=False)
tensorboard.set_model(model)
callbacks.append(tensorboard)

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks, \
                validation_data=val_generator, validation_steps=N_seq_val / batch_size, verbose=1)

# summarize history for loss
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('loss_full_kitti_ev_t_1.png')

# save history in a hickle file
hkl.dump(history.history, 'history_full_kitti_ev_t_1.hkl', mode='w')

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

