'''
Particle filter velocity estimates and probabilistic occupancy grids are processed into static and DOGMa grids that will serve as input into PredNet. 
This file assumes the full training tracking dataset is used.

Code modified from: https://github.com/coxlab/prednet
'''

import os
import numpy as np
import hickle as hkl
from kitti_settings import *

DATA_DIR = "../../Data/ParticleFilter/MahalanobisVelocityGrids/"
EVIDENTIAL_DIR = "../../Data/ParticleFilter/EvidentialGrids/"
OUTPUT_DIR_DATA = "../../Data/PredNetData/"
OUTPUT_DIR_DOGMA = "../../Data/PredNetData/DOGMa/"
OUTPUT_DIR_EVIDENTIAL = "../../Data/PredNetData/EvidentialGrids/"

# Recordings used for validation and testing.
val_recordings = ['2011_09_26_drive_0005_sync_0000000153.hkl','2011_09_26_drive_0014_sync_0000000313.hkl']
test_recordings = ['2011_09_26_drive_0091_sync_0000000339.hkl', '2011_09_26_drive_0015_sync_0000000296.hkl']

if not os.path.exists(OUTPUT_DIR_DATA): os.mkdir(OUTPUT_DIR_DATA)
if not os.path.exists(OUTPUT_DIR_DOGMA): os.mkdir(OUTPUT_DIR_DOGMA)
if not os.path.exists(OUTPUT_DIR_EVIDENTIAL): os.mkdir(OUTPUT_DIR_EVIDENTIAL)

# Create grid datasets.
# Processes grids and saves them in train, val, test splits.
def process_data():
	splits = {s: [] for s in ['train', 'test', 'val']}
	splits['val'] = val_recordings
	splits['test'] = test_recordings
	not_train = splits['val'] + splits['test']

	max_vel = np.NINF
	min_vel = np.Inf

	for fn in sorted(os.listdir(DATA_DIR)):

		if (fn[-3:] == 'hkl'):

			if (fn not in not_train):
				splits['train'] += [fn]
			
			DOGMA = np.transpose(hkl.load(os.path.join(DATA_DIR, fn)), (0,2,3,1))/3.
			DOGMA_max = np.amax(DOGMA)
			DOGMA_min = np.amin(DOGMA)

			if DOGMA_max > max_vel:
				max_vel = DOGMA_max
		
			if DOGMA_min < min_vel:
				min_vel = DOGMA_min		

	for split in splits:
		im_list = []
		source_list = []  # corresponds to recording that image came from

		found = False

		for name in splits[split]:

			# convert to m/s
			DOGMA = np.transpose(hkl.load(os.path.join(DATA_DIR, name)), (0,2,3,1)).astype('float16')/3.
			EVIDENTIAL = np.transpose(np.expand_dims(np.array(hkl.load(os.path.join(EVIDENTIAL_DIR, name))),axis=4).astype('float16'), (0,2,3,1,4))

			# change the label from car from 3 to 1 (occupied), normalize: 0 - {F,O}, 0.5 - {O}, 1.0 - {F}
			EVIDENTIAL[np.where(EVIDENTIAL==3.)] = 1.

			# combine probabilities with velocities
			grid = np.concatenate((EVIDENTIAL[:,:,:,2,:], DOGMA), axis=3)

			if not found:
				X = grid
				X_ev = EVIDENTIAL[:,:,:,2,:]
				found = True

			else:
				X = np.concatenate((X, grid), axis=0)
				X_ev = np.concatenate((X_ev, EVIDENTIAL[:,:,:,2,:]), axis=0)
			
			source_list += [name[:-15]] * EVIDENTIAL.shape[0]

			print 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images'

		
		# normalize the velocities to 0 - > 1
		max_vel = np.amax(X)
		min_vel = np.amin(X)
		
		# used to be 1, 2
		X[:,:,:,1] = (X[:,:,:,1] - min_vel)/(max_vel-min_vel) 
		X[:,:,:,2] = (X[:,:,:,2] - min_vel)/(max_vel-min_vel)
		
		hkl.dump(X, os.path.join(OUTPUT_DIR_DOGMA, 'X_' + split + '.hkl'))
		hkl.dump(source_list, os.path.join(OUTPUT_DIR_DOGMA, 'sources_' + split + '.hkl'))
		
		hkl.dump(X_ev, os.path.join(OUTPUT_DIR_EVIDENTIAL, 'X_' + split + '.hkl'))
		hkl.dump(source_list, os.path.join(OUTPUT_DIR_EVIDENTIAL, 'sources_' + split + '.hkl'))
		
if __name__ == '__main__':
    process_data()
