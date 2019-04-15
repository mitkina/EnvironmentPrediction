# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 09:40:51 2017

@author:

Main function for the static occupancy grid generation. Procedure followed mostly from:

D. Nuss, T. Yuan, G. Krehl, M. Stuebler, S. Reuter, and K. Dietmayer. Fusion of laser and
radar sensor data with a Sequential Monte Carlo Bayesian Occupancy Filter. IEEE Intelligent
Vehicles Symposium (IV), pages 1074–1081, 2015.

"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pdb
from scipy.stats import itemfreq
from copy import deepcopy
seed = 1987
np.random.seed(seed)

import pickle
import hickle as hkl
import time
import sys
import os
import math

from matplotlib import pyplot as plt

sys.path.insert(0, '..')
from PlotTools import colorwheel_plot, particle_plot

DATA_DIR = "../../Data/SensorMeasurementsEgo/"
OUTPUT_DIR = "../../Data/ParticleFilter/EvidentialGrids/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Populate the Dempster-Shafer measurement masses.
def create_DST_grids(grids, meas_mass=0.95):
    
    data = []
    prev_indeces = np.where(grids[0,:,:] == 3) 

    for j in range(grids.shape[0]):

        grid = grids[j,:,:]
        free_array = np.zeros(grid.shape)
        occ_array = np.zeros(grid.shape)
        
        # occupied indeces
        indeces = np.where(grid == 1)
        occ_array[indeces] = meas_mass

        # free indeces
        indeces = np.where(grid == 2)
        free_array[indeces] = meas_mass

        # car
        indeces = np.where(grid == 3)
        occ_array[indeces] = 1.

        data.append(np.stack((free_array, occ_array)))

    data = np.array(data)        
        
    return data

def main():
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn[-3:] == 'hkl':

            [grids, gridglobal_x, gridglobal_y, transforms, vel_east, vel_north, acc_x, acc_y, adjust_indices] = hkl.load(DATA_DIR + fn)

            minx_global = np.amin(gridglobal_x[:,0])
            maxx_global = np.amax(gridglobal_x[:,0])
            miny_global = np.amin(gridglobal_y[0,:])
            maxy_global = np.amax(gridglobal_y[0,:])
    
            grids = np.array(grids)
            print fn, grids.shape

            do_plot = True # Toggle me for DOGMA plots!

            # PARAMETERS
            alpha = 0.9                                           # information ageing (discount factor) - how much we discount old information
            res = 1./3.

            shape = grids.shape[1:]

            # print debug values
            verbose = True

            # index where PF was interrupted
            index_stopped = 0

            # data: [N x 2 x W x D]
            # second dimension is masses {0: m_free, 1: m_occ}
            # in original grid: 0: unknown, 1: occupied, 2: free (raw data)
            data = create_DST_grids(grids)
    
            # number of measurements in the run
            N = data.shape[0]
            print "shape", data.shape
    
            # list of 4x256x256 (or 6) grids with position, velocity information 
            Masses = []

            # run particle filter iterations
            for i in range(N):

                start = time.time()

                prev_free = np.zeros(grids.shape[1:])
                prev_occ = np.zeros(grids.shape[1:])
    
                # initializes a measurement cell array
                meas_free = data[i,0,:,:] 
                meas_occ = data[i,1,:,:]

                # compute the previous grids
                # get the local grids previous
                # get the current local grid

                # get the GPS coordinates
                centerpoint = np.array([transforms[i][0,3],transforms[i][1,3]])

                # center point coordinates within global grid
                indxc = find_nearest(gridglobal_x[:,0].shape[0],centerpoint[0],minx_global,maxx_global,res)
                indyc = find_nearest(gridglobal_y[0,:].shape[0],centerpoint[1],miny_global,maxy_global,res)

                # MAKES 128 x 128 grids 
                minx = indxc - int((128./2./3.)/res)
                miny = indyc - int((128./2./3.)/res)
                maxx = indxc + int((128./2./3.)/res)
                maxy = indyc + int((128./2./3.)/res)

                x_new_low = gridglobal_x[minx,0]
                x_new_high = gridglobal_x[maxx,0] 
                y_new_low = gridglobal_y[0,miny]
                y_new_high = gridglobal_y[0,maxy]

                if i > 0:

                    xstart = None
                    ystart = None

                    if ((x_new_low >= x_old_low) and (x_old_high >= x_new_low)):
                        xstart = x_new_low
                        xend = x_old_high

                    if ((y_new_low >= y_old_low) and (y_old_high >= y_new_low)):
                        ystart = y_new_low 
                        yend = y_old_high

                    if ((x_new_low < x_old_low) and (x_new_high >= x_old_low)):
                        xstart = x_old_low
                        xend = x_new_high

                    if ((y_new_low < y_old_low) and (y_new_high >= y_old_low)): 
                        ystart = y_old_low
                        yend = y_new_high                      

                    if ((xstart != None) and (ystart != None)):
    
                        # compute the previous grid
                        indx_nl = find_nearest(grids.shape[1],xstart,x_new_low,x_new_high,res)
                        indx_nh = find_nearest(grids.shape[1],xend,x_new_low,x_new_high,res)
                        indy_nl = find_nearest(grids.shape[2],ystart,y_new_low,y_new_high,res)
                        indy_nh = find_nearest(grids.shape[2],yend,y_new_low,y_new_high,res)

                        indx_ol = find_nearest(grids.shape[1],xstart,x_old_low,x_old_high,res)
                        indx_oh = find_nearest(grids.shape[1],xend,x_old_low,x_old_high,res)
                        indy_ol = find_nearest(grids.shape[2],ystart,y_old_low,y_old_high,res)
                        indy_oh = find_nearest(grids.shape[2],yend,y_old_low,y_old_high,res)
    
                        print indx_nl, indx_nh, indy_nl, indy_nh, indx_ol, indx_oh, indy_ol, indy_oh
                        print "xs", x_new_low, x_new_high, x_old_low, x_old_high
                        print "new x lims", xstart, xend
                        prev_free[indx_nl:(indx_nh+1),indy_nl:(indy_nh+1)] = deepcopy(up_free[indx_ol:(indx_oh+1), indy_ol:(indy_oh+1)])
                        prev_occ[indx_nl:(indx_nh+1),indy_nl:(indy_nh+1)] = deepcopy(up_occ[indx_ol:(indx_oh+1), indy_ol:(indy_oh+1)])

                # MassUpdate (stored in grid_cell_array)
                up_free, up_occ = mass_update(meas_free, meas_occ, prev_free, prev_occ, alpha)

                print "occupancy prediction complete"

                newMass = get_mass(up_free, up_occ, grids[i,:,:])

                # save the DOGMA at this timestep
                if (i+1) > index_stopped:
                    Masses.append(newMass)

                print "Masses saved"

                end = time.time()
                print "Time per iteration: ", end - start 

                # save the old grid boundaries
                x_old_low = deepcopy(x_new_low)
                x_old_high = deepcopy(x_new_high)
                y_old_low = deepcopy(y_new_low)
                y_old_high = deepcopy(y_new_high)

                if i == 65:
                    plt.matshow(newMass[2,:,:])
                    plt.savefig(OUTPUT_DIR + fn[0:-4] + '_' + str(i) + '.png', dpi=100)

                print "Iteration ", i, " complete"

            hkl.dump(Masses, os.path.join(OUTPUT_DIR, fn), mode='w')
            print "Masses written to hickle file."

    return

# for now only save occupied and free masses
"""Need to save measurement occupancy grid instead of just particle occupancies (or in addition)!"""
def get_mass(up_free, up_occ, meas_grid):

    probO = 0.5*up_occ + 0.5*(1.-up_free)
    newMass = np.stack((up_occ, up_free, probO, meas_grid))

    return newMass

def mass_update(meas_free, meas_occ, prev_free, prev_occ, alpha):
            
        check_values = False

        # predicted mass
        m_occ_pred = np.minimum(alpha * prev_occ, 1. - prev_free)
        m_free_pred = np.minimum(alpha * prev_free, 1. - prev_occ)

        if check_values and (m_occ_pred > 1 or m_occ_pred < 0):
            if m_occ_pred > 1.:
                print "This is m_occ_pred: ", m_occ_pred
            assert(m_occ_pred <= 1.)
            assert (m_occ_pred >= 0.)
            assert (m_free_pred <= 1. and m_free_pred >= 0.)
            assert (m_occ_pred + m_free_pred <= 1.)

        # combine measurement and prediction to form posterior occupied and free masses
        m_occ_up, m_free_up = update_of(m_occ_pred, m_free_pred, meas_occ, meas_free)

        if check_values and (m_occ_up > 1.001 or m_occ_up < 0.):
            print "mass_occ: ", m_occ_up, "mass_free: ", m_free_up
            assert(m_occ_up <= 1. and m_occ_up >= 0.)
            assert (m_free_up <= 1. and m_free_up >= 0.)
            assert(m_occ_up + m_free_up <= 1.)
   
        return m_free_up, m_occ_up

# equation 63: perform dst update
def update_of(m_occ_pred, m_free_pred, meas_m_occ, meas_m_free):
    
    # predicted unknown mass
    m_unknown_pred = 1. - m_occ_pred - m_free_pred
    
    # measurement masses: meas_m_free, meas_m_occ
    meas_cell_unknown = 1. - meas_m_free - meas_m_occ
    
    # implement DST rule of combination
    K = np.multiply(m_free_pred, meas_m_occ) + np.multiply(m_occ_pred, meas_m_free)
    
    m_occ_up = np.divide((np.multiply(m_occ_pred, meas_cell_unknown) + np.multiply(m_unknown_pred, meas_m_occ) + np.multiply(m_occ_pred, meas_m_occ)), (1. - K))
    m_free_up = np.divide((np.multiply(m_free_pred, meas_cell_unknown) + np.multiply(m_unknown_pred, meas_m_free) + np.multiply(m_free_pred, meas_m_free)), (1. - K))    
    
    return m_occ_up, m_free_up

def find_nearest(n,v,v0,vn,res):
    "Element in nd array closest to the scalar value `v`"
    idx = int(np.floor( n*(v-v0+res/2.)/(vn-v0+res) ))
    return idx


if __name__ == "__main__":
    main()
