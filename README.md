# EnvironmentPrediction
Implementation of occupancy grid predictions via ConvLSTM architecture as in:

Itkina, Masha, Katherine Driggs-Campbell, and Mykel J. Kochenderfer. "Dynamic Environment Prediction in Urban Scenes using Recurrent Representation Learning." arXiv preprint arXiv:1904.12374 (2019).

The attached code demonstrates the process of LiDAR data pre-processing into occupancy grids and DOGMas, and then portrays the training on the PredNet architecture.

To begin, please download the KITTI tracking dataset from: http://www.cvlibs.net/datasets/kitti/raw_data.php. Copy the resulting directories into the Data/KITTI/ subdirectory.

Run the following scripts from the top directory:

'run_sensor_measurements.sh'
'run_grid_generation.sh'
'run_predictions.sh'

This sequence of commands runs ground segmentation, occupancy grid and DOGMa generation, PredNet training in t+1 then t+5 mode on DST and probabilistic static grid and DOGMa data, and evaluation on the test set. The input data and results are located in the 'Data/' directory.

The code is written in Python 2.7 with the following pip dependencies:

hickle==2.1.0

Keras==2.0.6

numpy==1.13.3

tensorboard==1.6.0

tensorflow-gpu==1.6.0

tensorflow-tensorboard==0.4.0rc3
