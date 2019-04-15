# EnvironmentPrediction
Implementation of occupancy grid predictions via ConvLSTM video frame prediction architecture.

The attached code demonstrates the process of LiDAR data pre-processing into occupancy grids and DOGMas, and then portrays the training on the PredNet architecture.

To begin, please download the KITTI tracking dataset from: http://www.cvlibs.net/datasets/kitti/raw_data.php. Copy the resulting directories into the Data/KITTI/ subdirectory.

Run the 'run_all.sh' script from the main directory, which contains all the necessary commands in sequence. This sequence of commands runs ground segmentation, occupancy grid and DOGMa generation, PredNet training in t+1 then t+5 mode on DST DOGMa data, and evaluation on the test set.

The code is written in Python 2.7 with the following pip dependencies:

hickle==2.1.0
Keras==2.0.6
numpy==1.13.3
tensorboard==1.6.0
tensorflow-gpu==1.6.0
tensorflow-tensorboard==0.4.0rc3
