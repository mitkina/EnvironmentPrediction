# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:51:08 2017

@author: Masha Itkina

Generated sensor measurement grids from KITTI LiDAR data.

Data imported from KITTI: http://www.cvlibs.net/datasets/kitti/raw_data.php
(note only synced data)

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

# loop through the training sequences, save each set of data separately
KITTI_DIR = '../../Data/KITTI/'
OUTPUT_DIR = '../../Data/SensorMeasurements/'

if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

# 2D grid resolution
res = 1./3.
i = 0

for folder_main in sorted(os.listdir(KITTI_DIR)):
	print folder_main

	if os.path.isfile(os.path.join(KITTI_DIR, folder_main)):
		continue

	for folder_second in sorted(os.listdir(os.path.join(KITTI_DIR, folder_main))):
		print folder_second

		if ((folder_second[-4:] == 'sync')):
			print folder_second

			VELODYNE_DIR = os.path.join(KITTI_DIR, folder_main, folder_second, 'velodyne_points/data/')
			OXTS_DIR = os.path.join(KITTI_DIR, folder_main, folder_second, 'oxts/data/')

			print "sizes", len(sorted(os.listdir(VELODYNE_DIR))), len(sorted(os.listdir(OXTS_DIR)))

			# initialize the list of transformations
			transforms = []
			# Scale for Mercator projection (from first lat value)
			scale = None
			# Origin of the global coordinate system (first GPS position)
			origin = None
			# find the minimum gps point
			xminGPS = 10**12
			yminGPS = 10**12
			xmaxGPS = -10**12
			ymaxGPS = -10**12

			vel_east = []
			vel_north = []
			acc_x = []
			acc_y = []
			
			# OXTS pre-processing is from pykitti: https://github.com/utiasSTARS/pykitti/
			for fn_oxts in sorted(os.listdir(OXTS_DIR)):
				# print fn_oxts

				sequence = folder_second + '_' + fn_oxts[0:10]

				# load GPS position data and velocity
				scan = open(OXTS_DIR + fn_oxts,'r')

				for line in scan.readlines():
		
					# lat:     latitude of the oxts-unit (deg)
					# lon:     longitude of the oxts-unit (deg)
					# alt:     altitude of the oxts-unit (m)
					# roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
					# pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
					# yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
					# vn:      velocity towards north (m/s)
					# ve:      velocity towards east (m/s)
					# vf:      forward velocity, i.e. parallel to earth-surface (m/s)
					# vl:      leftward velocity, i.e. parallel to earth-surface (m/s)
					# vu:      upward velocity, i.e. perpendicular to earth-surface (m/s)
					# ax:      acceleration in x, i.e. in direction of vehicle front (m/s^2)
					# ay:      acceleration in y, i.e. in direction of vehicle left (m/s^2)
					# az:      acceleration in z, i.e. in direction of vehicle top (m/s^2)
					# af:      forward acceleration (m/s^2)
					# al:      leftward acceleration (m/s^2)
					# au:      upward acceleration (m/s^2)
					# wx:      angular rate around x (rad/s)
					# wy:      angular rate around y (rad/s)
					# wz:      angular rate around z (rad/s)
					# wf:      angular rate around forward axis (rad/s)
					# wl:      angular rate around leftward axis (rad/s)
					# wu:      angular rate around upward axis (rad/s)
					# posacc:  velocity accuracy (north/east in m)
					# velacc:  velocity accuracy (north/east in m/s)
					# navstat: navigation status
					# numsats: number of satellites tracked by primary GPS receiver
					# posmode: position mode of primary GPS receiver
					# velmode: velocity mode of primary GPS receiver
					# orimode: orientation mode of primary GPS receiver

					# split the line into a list of string numbers 
					line = line.split()
					# the last 5 elements of each line are counts and states (integers)
					line[:-5] = [float(x) for x in line[:-5]]
					line[-5:] = [int(float(x)) for x in line[-5:]]
					[lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, ax, ay, az, af, al, au, wx, wy, wz, wf, wl, wu, posacc, velacc] = line[:-5]
		
					vel_east.append(ve)
					vel_north.append(vn)
					acc_x.append(ax)
					acc_y.append(ay)

					# Mercator projection starts from the first point
					if scale == None:
						# use the latitude to scale
						scale = np.cos(line[0] * np.pi / 180.)

					R, t = pose_from_oxts_packet(line[0],line[1],line[2],line[3],line[4],line[5], scale)

					# check if origin is already set
					if origin is None:
						origin = t
		
					# create the homogeneous transformation from the coordinates
					T_w_imu = transform_from_rot_trans(R, t - origin)

					# keep track of the edge x, y coordinates for the map
					if T_w_imu[0,3] > xmaxGPS:
						xmaxGPS = T_w_imu[0,3]
					if T_w_imu[0,3] < xminGPS:
						xminGPS = T_w_imu[0,3]
					if T_w_imu[1,3] > ymaxGPS:
						ymaxGPS = T_w_imu[1,3]
					if T_w_imu[1,3] < yminGPS:
						yminGPS = T_w_imu[1,3]

					# append the homogeneous transformation matrix to the list
					transforms.append(T_w_imu)

					# if i%10 == 0:
					# 	print("xmin: ", xminGPS, "xmax: ", xmaxGPS)

					i += 1

			print "transforms length", len(transforms), i

			# load the training dataset (velodyne)
			# the frame will be relative to the first GPS point
			data = []
			# initialize a counter 0
			counter = 0
			for fn in sorted(os.listdir(VELODYNE_DIR)):
				# print fn
				start = time.time()

				scan = np.fromfile(VELODYNE_DIR + fn, dtype=np.float32)
		
				# reshape velodyne data as [x,y,z,reflectance]
				new = scan.reshape((-1, 4))

				# perform ground segmentation according to Postica's Markov Random Field Formulation
				new = ground_seg(new[:,:3], res=0.4, s=0.09)
					                    
				# transform each point to global coordinates using GPS
				# transform from velodyne to GPS

				H_GPS_velo = np.array(([1.,0.,0.,0.81],[0.,1.,0.,-0.32],[0.,0.,1.,-(0.93-1.73)],[0.,0.,0.,1.]))
				new2 = np.vstack((new[:,:].T,np.ones((1,new.shape[0]))))
				new2 = transforms[counter].dot(H_GPS_velo.dot(new2))
				new = new2[0:3,:]

				# get the global coordinates of the car
				centerpoint = np.array([transforms[counter][0,3],transforms[counter][1,3]])

				# get the coordinates of the velodyne in the global frame
				velx = centerpoint[0]+0.81
				vely = centerpoint[1]-0.32  

				# ensure the points are outside the car (min radius 0.8 m)
				distances3d = np.linalg.norm(np.vstack((new[0,:],new[1,:]))-np.vstack((np.reshape(np.array([velx,vely]),(2,1)))),axis=0)
				new = np.delete(new,np.where(distances3d  < 0.8),axis=1)

				data.append(new.T)

				counter += 1
				print counter, time.time()-start

			endpoint = transforms[-1][0:2,3]

			gridglobal_x,gridglobal_y = global_grid(np.array([xminGPS,yminGPS]),np.array([xmaxGPS,ymaxGPS]),res)  

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
			sensorgrid = [] 
			adjust_indices = []

			for i in range(len(data)):

				start_total = time.time()
				# print(i)

				# get the GPS coordinates
				centerpoint = np.array([transforms[i][0,3],transforms[i][1,3]])

				# center point coordinates within global grid
				indxc = find_nearest(x_shape_global,centerpoint[0],x_min_global,x_max_global,res) # gridglobal_x[:,0],centerpoint[0])
				indyc = find_nearest(y_shape_global,centerpoint[1],y_min_global,y_max_global,res) # (gridglobal_y[0,:],centerpoint[1])

				# compute the local grid transformation
				if i != 0:
					x_adjust = indxc - indxc_prev
					y_adjust = indyc - indyc_prev

					adjust_indices.append([x_adjust, y_adjust])		

				# label the lines along the ray as free
				# transform the GPS coord to velodyne coord
				velx = centerpoint[0]+0.81
				vely = centerpoint[1]-0.32
				pointcloud = data[i]

				# truncate the newgrid to only center around GPS (42.7 m in each direction)
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

			    # create an empty grid
				newgrid = np.zeros(gridglobal_x[minx:(maxx),miny:(maxy)].shape) # zero is unknown

				#print 'Label the ego vehicle', time.time() - start

				gridlocal_x = gridglobal_x[minx:(maxx),miny:(maxy)]
				gridlocal_y = gridglobal_y[minx:(maxx),miny:(maxy)]
				x_max_local = np.amax(gridlocal_x[:,0])
				y_max_local = np.amax(gridlocal_y[0,:])

				x_min_local = np.amin(gridlocal_x[:,0])
				y_min_local = np.amin(gridlocal_y[0,:])
				x_shape_local = gridlocal_x.shape[0]
				y_shape_local = gridlocal_y.shape[1]

				start = time.time()
				for l in range(len(pointcloud)):
					datapoint = pointcloud[l]
					x = datapoint[0]
					y = datapoint[1]

					if ((x > (x_max_local + res/2.)) or (x < (x_min_local - res/2.)) or (y > (y_max_local + res/2.)) or (y < (y_min_local))):
						continue

					indx = find_nearest(x_shape_local,x,x_min_local,x_max_local,res)
					indy = find_nearest(y_shape_local,y,y_min_local,y_max_local,res)
	
					# indicate as occupied
					newgrid[indx,indy] = 1
		
				#print 'Mark all cells as occupied:', start-time.time()
		
				xind,yind = np.where(newgrid==1)

				# compute the distances from each of the obstacles to the velodyne center
				xs = gridlocal_x[xind,yind]
				ys = gridlocal_y[xind,yind]
				distances = np.linalg.norm(np.vstack((np.reshape(xs,(1,xs.shape[0])),np.reshape(ys,(1,ys.shape[0]))))-np.vstack((np.reshape(np.array([velx,vely]),(2,1)))),axis=0)
		
				# ensure the points are outside the car (min radius 0.8 m)
				xind = np.delete(xind, np.where(distances < 0.8))
				yind = np.delete(yind, np.where(distances < 0.8))
				distances = np.delete(distances,np.where(distances < 0.8))
		
				xind = np.delete(xind, np.where(distances > np.sqrt(2.*(42.7/2.)**2)))
				yind = np.delete(yind, np.where(distances > np.sqrt(2.*(42.7/2.)**2)))
				distances = np.delete(distances,np.where(distances > np.sqrt(2.*(42.7/2.)**2)))

				newlist = np.argsort(distances)
				xind = xind[newlist]
				yind = yind[newlist]

				start = time.time()
		
				for k in range(xind.shape[0]):

					# create a line of x's spaced 0.33 m apart (leave a 1.0 m buffer to 
					# get out of the space of the car)
					x = gridlocal_x[xind[k],yind[k]]
					y = gridlocal_y[xind[k],yind[k]]
	
					if x < velx:
						x_range = np.arange(velx,x,-res*0.01) # 0.05
					else:
						x_range = np.arange(velx,x,res*0.01) # 0.05
					# find the corresponding ys
					yrange = linefunction(velx,vely,x,y,x_range)

					# loop through each segment, and find the closest coordinates
					for j in range(x_range.shape[0]): # move from the inside -> out

						xtemp = find_nearest(x_shape_local,x_range[j],x_min_local,x_max_local,res)
						ytemp = find_nearest(y_shape_local,yrange[j],y_min_local,y_max_local,res)

						if newgrid[xtemp,ytemp] == 1:
							break
						newgrid[xtemp,ytemp] = 2      

				print 'fill in the unknown regions', time.time() - start

				# label the ego vehicle 
				x_ahead = transforms[i].dot(H_GPS_velo.dot(np.array([0.27 + 1.68,0.8,0.,1.])))[:2]
				x_behind = transforms[i].dot(H_GPS_velo.dot(np.array([-0.81,-0.8,0.,1.])))[:2]
				y_ahead = transforms[i].dot(H_GPS_velo.dot(np.array([-0.81,0.8,0.,1.])))[:2]
				y_behind = transforms[i].dot(H_GPS_velo.dot(np.array([0.27 + 1.68,-0.8,0.,1.])))[:2]
				ego = np.vstack((x_behind,y_ahead,x_ahead,y_behind))

				start = time.time()

				for x in range(newgrid.shape[0]):
					for y in range(newgrid.shape[1]):
						point = np.array([gridlocal_x[x,y], gridlocal_y[x,y]])
						if point_in_polygon(point,ego,float('Inf')):
							newgrid[x,y] = 3

				# update the previous global grid center
				indxc_prev = copy.deepcopy(indxc)
				indyc_prev = copy.deepcopy(indyc)
		
				sensorgrid.append(newgrid)   

				print 'End of one scan', time.time()-start_total

			print "one run completed", len(sensorgrid)
			if (len(sensorgrid) > 0):
				# local velodyne
				fig1 = plt.figure()
				ax1 = fig1.add_subplot(111)
				h = ax1.contourf(gridglobal_x[minx:(maxx),miny:(maxy)],gridglobal_y[minx:(maxx),miny:(maxy)],sensorgrid[-1])
				plt.colorbar(h)
				#plt.show()
				fig1.savefig(OUTPUT_DIR + 'velodynelocal_example_' + sequence + '.png') 

				# Saving the objects:
				hkl.dump([sensorgrid, gridglobal_x, gridglobal_y, transforms, vel_east, vel_north, acc_x, acc_y, adjust_indices], OUTPUT_DIR + sequence+'.hkl', mode='w')

