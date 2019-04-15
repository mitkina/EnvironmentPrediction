#! /bin/bash

cd Predictions/SensorMeasurements/
python sensor_measurement_generation_full_kitti.py
python sensor_measurement_ego_correction_full_kitti.py