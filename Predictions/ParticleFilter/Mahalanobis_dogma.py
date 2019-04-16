# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 09:40:51 2017

@author: Masha

Filter the DOGMA data based on Mahanobis distance (preprocessing step).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pdb
from scipy.stats import itemfreq

seed = 1987
np.random.seed(seed)

import hickle as hkl
import time
import sys
import os
import math
sys.path.insert(0, '..')
from PlotTools import colorwheel_plot, particle_plot
import pdb

DATA_DIR = "../../Data/ParticleFilter/VelocityGrids/"
OUTPUT_DIR = "../../Data/ParticleFilter/MahalanobisVelocityGrids/"

if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

def main():

	for fn in sorted(os.listdir(DATA_DIR)):

		if (fn[-3:] == 'hkl'):

			OUTPUT_DIR_IMAGES = OUTPUT_DIR + fn[0:-4] + '/'
			if not os.path.exists(OUTPUT_DIR_IMAGES):
				os.makedirs(OUTPUT_DIR_IMAGES)

			print fn

			[DOGMA,var_x_vel, var_y_vel, covar_xy_vel] = hkl.load(os.path.join(DATA_DIR, fn))

			# posO,posF,velX,velY,meas_grid 
			DOGMA = np.array(DOGMA)	
			var_x_vel = np.array(var_x_vel)
			var_y_vel = np.array(var_y_vel)
			covar_xy_vel = np.array(covar_xy_vel)

			do_plot = True # Toggle me for DOGMA plots!
	
			# velocity, acceleration variance initialization
			scale_vel = 12.
			scale_acc = 2.

			# position, velocity, acceleration process noise
			process_pos = 0.06
			process_vel = 2.4
			process_acc = 0.2

			# for plotting thresholds
			mS = 4. # 3.         # 4. static threshold
			epsilon = 10.   # vel mag threshold
			epsilon_occ = 0.95 # 0.75 # occ mag threshold
	
			# number of measurements in the run
			N = DOGMA.shape[0]

			newDOGMA = mahalanobis_filter(DOGMA, var_x_vel, var_y_vel, covar_xy_vel, mS, epsilon, epsilon_occ)

			print newDOGMA.shape
			
			if not os.path.exists(OUTPUT_DIR):
				os.makedirs(OUTPUT_DIR)
			
			hkl.dump(newDOGMA, os.path.join(OUTPUT_DIR + fn), mode="w")

			if do_plot:
				for i in range(N):

					# Plotting: The environment is stored in grids[i] (matrix of  values (0,1,2))
					#           The DOGMA is stored in DOGMA[i]
					head_grid = dogma2head_grid(newDOGMA[i,:,:,:], DOGMA[i,0,:,:], var_x_vel[i], var_y_vel[i], covar_xy_vel[i], mS, epsilon, epsilon_occ)
					occ_grid = DOGMA[i,4,:,:]
					title = str(i) #"DOGMa Sequence %s Iteration %d" % (fn[0:5], i)
					colorwheel_plot(head_grid, occ_grid=occ_grid, m_occ_grid = DOGMA[i,0,:,:], title=os.path.join(OUTPUT_DIR_IMAGES, title), \
								show=True, save=True)

					print "Iteration ", i, " complete"

	return

def mahalanobis_filter(dogma, var_x_vel, var_y_vel, covar_xy_vel, mS = 4., epsilon=0.5, epsilon_occ=0.1):
    """Create a filtered dogma according to Mahalanobis distance.
    """
    print dogma.shape
    grid_shape = (dogma.shape[0], 2, dogma.shape[2], dogma.shape[3])
    filtered_dogma = np.zeros(grid_shape)
    mdist = np.zeros((dogma.shape[0], dogma.shape[2], dogma.shape[3]))

    vel_x = dogma[:,2,:,:]
    vel_y = dogma[:,3,:,:]
    m_occ = dogma[:,0,:,:]
    m_free = dogma[:,1,:,:]

    # mahalanobis distance
    covar = np.transpose(np.array([[var_x_vel, covar_xy_vel], [covar_xy_vel, var_y_vel]]),(2,3,4,0,1))
    mask = np.absolute(np.linalg.det(covar)) < 10**(-6)
    mdist[mask] = 0.
    vels = np.transpose(np.expand_dims(np.array([vel_x[np.logical_not(mask)], vel_y[np.logical_not(mask)]]), axis=1), (2,1,0))
    inv_cov = np.transpose(np.linalg.inv(covar[np.logical_not(mask),:,:]),(0,1,2))
    intermediate = np.matmul(vels, inv_cov)
    mdist[np.logical_not(mask)] = np.squeeze(np.matmul(intermediate, np.transpose(vels,(0,2,1))))

    mag = np.sqrt(np.square(vel_x) + np.square(vel_y))
    # occupied and with velocity
    index_0, index_1, index_2 = np.where(( np.logical_and( (mdist > mS), (m_occ > epsilon_occ) ) )) # (mag > epsilon) # and (var_x_vel[i,j] < 27.) and (var_y_vel[i,j] < 27.) and (np.sqrt(vel_x[i,j]**2 + vel_y[i,j]**2) > 2.1)
    filtered_dogma[index_0,:,index_1,index_2] = dogma[index_0,2:4,index_1,index_2]

    return filtered_dogma

def dogma2head_grid(dogma, m_occ, var_x_vel, var_y_vel, covar_xy_vel, mS = 4., epsilon=0.5, epsilon_occ=0.1):
    """Create heading grid for plotting tools from a DOGMA.
    USAGE:
        head_grid = dogma2head_grid(dogma, (epsilon) )
    INPUTS:
        dogma - (np.ndarray) Single DOGMA tensor (supports size of 4 or 6)
        epsilon - (opt)(float) Minimum cell vel mag required to plot heading
    OUTPUTS:
        head_grid - (np.matrix) Grid (of same shape as each vel grid) containing
                                object headings at each cell, in rad
    """
    # Initialize grid with None's; this distinguishes from a 0rad heading!
    head_grid = np.full((dogma.shape[1], dogma.shape[2]), None, dtype=float)
    vel_x = dogma[0,:,:]
    vel_y = dogma[1,:,:]
    # Fill grid with heading angles where we actually have velocity
    mask = np.logical_and(np.logical_or((dogma[0,:,:] != 0.),(dogma[1,:,:] != 0.)), (m_occ > epsilon_occ))
    head_grid[mask] = np.arctan2(vel_y[mask], vel_x[mask])
    return head_grid

if __name__ == "__main__":
    main()
