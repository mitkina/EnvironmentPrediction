import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
import re
import hickle as hkl
from scipy import *

# reshape list
def reshape(seq, rows, cols):
    return [list(u) for u in zip(*[iter(seq)] * cols)]
    
# helper function from pykitti: https://github.com/utiasSTARS/pykitti
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

# helper function from pykitti: https://github.com/utiasSTARS/pykitti
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

# helper function from pykitti: https://github.com/utiasSTARS/pykitti
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

# helper function from pykitti: https://github.com/utiasSTARS/pykitti
def pose_from_oxts_packet(lat,lon,alt,roll,pitch,yaw,scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t

# position of vertex relative to global origin from pykitti: https://github.com/utiasSTARS/pykitti
def pose_from_GIS(lat,lon,scale,origin):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + lat) * np.pi / 360.))
    # 2D position
    t = np.array([tx, ty])

    return (t-origin[0:2])

# helper function from pykitti: https://github.com/utiasSTARS/pykitti
def transform_from_rot_trans(R, t):
    """Homogeneous transformation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

# functions to check if point is in polygon (http://stackoverflow.com/questions/16625507/python-checking-if-point-is-inside-a-polygon)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def point_in_polygon(pt, poly, inf):
    result = False
    for i in range(poly.shape[0]-1):
        if intersect((poly[i,0], poly[i,1]), ( poly[i+1,0], poly[i+1,1]), (pt[0], pt[1]), (inf, pt[1])):
            result = not result
    if intersect((poly[-1,0], poly[-1,1]), (poly[0,0], poly[0,1]), (pt[0], pt[1]), (inf, pt[1])):
        result = not result
    return result

# create a grid in the form of a numpy array with coordinates representing
# the middle of the cell
# cell resolution: 0.33 cm

def global_grid(origin,endpoint,res):
    # create a grid of x and y values that have an associated x,y centre 
    # coordinate in the global space (42.7 m buffer around the edge points)
    xmin = min(origin[0],endpoint[0])-(128./3.)
    xmax = max(origin[0],endpoint[0])+(128./3.)+res
    ymin = min(origin[1],endpoint[1])-(128./3.)
    ymax = max(origin[1],endpoint[1])+(128./3.)+res
    x_coords = np.arange(xmin,xmax,res)
    y_coords = np.arange(ymin,ymax,res)
    gridx,gridy = np.meshgrid(x_coords,y_coords)
  
    return gridx.T,gridy.T

def find_nearest(n,v,v0,vn,res):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = int(np.floor( n*(v-v0+res/2.)/(vn-v0+res) ))
    return idx

# generate the y indices along a line
def linefunction(velx,vely,indx,indy,xrange):
    m = (indy-vely)/(indx-velx)
    b = vely-m*velx
    return m*xrange + b 

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    for i in range(max_iterations): 
        s = data[np.random.choice(data.shape[0], 3, replace=False), :]
        m = estimate(s)
        ic = 0
        for j in range(data.shape[0]):
            if is_inlier(m, data[j,:]):
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print ('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic

def augment(xyzs):
	axyz = np.ones((len(xyzs), 4))
	axyz[:, :3] = xyzs
	return axyz

def estimate(xyzs):
	axyz = augment(xyzs[:3])
	return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
	return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def is_close(a,b,c,d,point,distance=0.1):
    D = (a*point[:,0]+b*point[:,1]+c*point[:,2]+d)/np.sqrt(a**2+b**2+c**2)
    return D

