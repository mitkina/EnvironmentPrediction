# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:51:08 2017

@author: Masha Itkina

Added ego vehicle to the sensor grids.

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import *

import os
import hickle as hkl
import copy
import time

from utils import *
from GroundSeg import *

import pdb
from copy import deepcopy

# loop through the training sequences, save each set of data separately
OUTPUT_DIR = '../../Data/SensorMeasurements/'
OUTPUT_DIR_EGO = '../../Data/SensorMeasurementsEgo/'
OUTPUT_DIR_EGO_IMAGES = '../../Data/SensorMeasurementsEgo/Images/'

if not os.path.exists(OUTPUT_DIR_EGO):
	os.makedirs(OUTPUT_DIR_EGO)

if not os.path.exists(OUTPUT_DIR_EGO_IMAGES):
	os.makedirs(OUTPUT_DIR_EGO_IMAGES)

# 2D grid resolution
res = 1./3.
i = 0

found = False

for fn in sorted(os.listdir(OUTPUT_DIR)):
	print fn

	if fn[-3:] == 'hkl':
		
		sequence = fn[:-4]

		IMAGES_FN = os.path.join(OUTPUT_DIR_EGO_IMAGES, sequence + '/')

		if not os.path.exists(IMAGES_FN):
			os.makedirs(IMAGES_FN)

		# Loading the objects:
		[sensorgrid, gridglobal_x, gridglobal_y, transforms, vel_east, vel_north, acc_x, acc_y, adjust_indices] = hkl.load(OUTPUT_DIR + fn)
	
		print "transforms length", len(transforms), i, len(sensorgrid), len(vel_east)
				                    
		# transform each point to global coordinates using GPS
		# transform from velodyne to GPS

		H_GPS_velo = np.array(([1.,0.,0.,0.81],[0.,1.,0.,-0.32],[0.,0.,1.,-(0.93-1.73)],[0.,0.,0.,1.]))

		x_min_global = np.amin(gridglobal_x[:,0])
		x_max_global = np.amax(gridglobal_x[:,0])
		y_min_global = np.amin(gridglobal_y[0,:])
		y_max_global = np.amax(gridglobal_y[0,:])

		x_shape_global = gridglobal_x.shape[0]
		y_shape_global = gridglobal_y.shape[1]     
				       
		# go through all the GPS locations, and create an occupancy grid for each
		# velodyne point cloud
		# 0 - {F,O}
		# 1 - {O}
		# 2 - {F}
		# 3 - ego vehicle
		# Implement a DST combination to show how many lidar hits for a single cell

		for i in range(len(sensorgrid)):

			start_total = time.time()
			# print(i)

			# get the GPS coordinates
			centerpoint = np.array([transforms[i][0,3],transforms[i][1,3]])

			# center point coordinates within global grid
			indxc = find_nearest(x_shape_global,centerpoint[0],x_min_global,x_max_global,res) # gridglobal_x[:,0],centerpoint[0])
			indyc = find_nearest(y_shape_global,centerpoint[1],y_min_global,y_max_global,res) # (gridglobal_y[0,:],centerpoint[1])

			# label the lines along the ray as free
			# transform the GPS coord to velodyne coord
			velx = centerpoint[0]+0.81
			vely = centerpoint[1]-0.32

			# truncate the grid to only center around GPS (42.7 m in each direction)
			minx = indxc - int((128./2./3.)/res) # int((128./3.)/res)
			miny = indyc - int((128./2./3.)/res) # int((128./3.)/res)
			maxx = indxc + int((128./2./3.)/res) # int((128./3.)/res)
			maxy = indyc + int((128./2./3.)/res) # int((128./3.)/res)

			# make sure within bounds of the global_grid
			if minx < 0:
				minx = 0
			if maxx >= (gridglobal_x.shape[0]-1):
				maxx = (gridglobal_x.shape[0]-1)
			if miny < 0:
				miny = 0
			if maxy >= (gridglobal_y.shape[1]-1):
				maxy = (gridglobal_y.shape[1]-1)

			# label the ego vehicle 
			x_ahead = transforms[i].dot(H_GPS_velo.dot(np.array([0.27 + 1.68,0.8,0.,1.])))[:2]
			x_behind = transforms[i].dot(H_GPS_velo.dot(np.array([-0.81,-0.8,0.,1.])))[:2]
			y_ahead = transforms[i].dot(H_GPS_velo.dot(np.array([-0.81,0.8,0.,1.])))[:2]
			y_behind = transforms[i].dot(H_GPS_velo.dot(np.array([0.27 + 1.68,-0.8,0.,1.])))[:2]
			ego = np.vstack((x_behind,y_ahead,x_ahead,y_behind))

			gridlocal_x = gridglobal_x[minx:(maxx),miny:(maxy)]
			gridlocal_y = gridglobal_y[minx:(maxx),miny:(maxy)]

			start = time.time()

			for x in range(sensorgrid[i].shape[0]):
				for y in range(sensorgrid[i].shape[1]):
					point = np.array([gridlocal_x[x,y], gridlocal_y[x,y]])
					if point_in_polygon(point,ego,float('Inf')):
						sensorgrid[i][x,y] = 3

			print 'End of one scan', time.time()-start_total

			if (len(sensorgrid) > 0):
				# local velodyne
				fig1 = plt.figure()
				ax1 = fig1.add_subplot(111)
				h = ax1.contourf(gridglobal_x[minx:(maxx),miny:(maxy)],gridglobal_y[minx:(maxx),miny:(maxy)],sensorgrid[i])
				plt.colorbar(h)
				#plt.show()
				fig1.savefig(IMAGES_FN + str(i) + '.png') 
				plt.close()

		print "one run completed", len(sensorgrid)
		
		# Saving the objects:
		hkl.dump([sensorgrid, gridglobal_x, gridglobal_y, transforms, vel_east, vel_north, acc_x, acc_y, adjust_indices], OUTPUT_DIR_EGO + fn, mode='w')

