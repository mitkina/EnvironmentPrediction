cd Predictions/ConvLSTMPredictions/

# Combine occupancy grids and velocity estimates into DST DOGMa and static grids.
python process_kitti_dst.py

# Combine occupancy grids and velocity estimates into Prob DOGMa and static grids.
python process_kitti.py

# EV TRAIN

# train PredNet in t+1 mode
python kitti_train_ev_t_1.py

# train PredNet in t+5 mode
python kitti_train_ev_t_5.py

# train PredNet in t+1 mode
python kitti_train_ev_t_1_dst.py

# train PredNet in t+5 mode
python kitti_train_ev_t_5_dst.py

# DOGMA TRAIN

# train PredNet in t+1 mode
python kitti_train_dogma_t_1.py

# train PredNet in t+5 mode
python kitti_train_dogma_t_5.py

# train PredNet in t+1 mode
python kitti_train_dogma_t_1_dst.py

# train PredNet in t+5 mode
python kitti_train_dogma_t_5_dst.py

# EV EVALUATE

# train PredNet in t+5 mode
python kitti_evaluate_ev_t_5.py

# train PredNet in t+5 mode
python kitti_evaluate_ev_t_5_dst.py

# DOGMA EVALUATE

# train PredNet in t+5 mode
python kitti_evaluate_dogma_t_5.py

# train PredNet in t+5 mode
python kitti_evaluate_dogma_t_5_dst.py