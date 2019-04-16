'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.

Code modified from: https://github.com/coxlab/prednet
'''

import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(123)
rn.seed(123)
from keras import backend as K
tf.set_random_seed(123)

import os
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

import hickle as hkl

import time
from timeit import default_timer as timer

n_plot = 40
batch_size = 1
nt = 20

# Data files
DATA_DIR_OCC = "../../Data/PredNetData/DOGMa/"
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_full_kitti_weights_dogma_t_5.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_full_kitti_model_dogma_t_5.json')

test_file = os.path.join(DATA_DIR_OCC, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR_OCC, 'sources_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = 5
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
start = timer()
X_hat = test_model.predict(X_test, batch_size)
end = timer()-start

RESULTS_SAVE_DIR_OCC = RESULTS_SAVE_DIR + "DOGMa/"
if not os.path.exists(RESULTS_SAVE_DIR_OCC): os.mkdir(RESULTS_SAVE_DIR_OCC)

# save history in a hickle file
hkl.dump([X_test, X_hat], RESULTS_SAVE_DIR_OCC + 'results_test.hkl', mode='w')

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2.*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR_OCC, 'plots_test/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t,:,:,0])
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t,:,:,0])

        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
    plt.clf()