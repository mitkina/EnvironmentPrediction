cd Predictions/ParticleFilter/

# Compute the occupancy grids
python EvidentialGridsFullKITTI.py

# Run Particle Filter
python ParticleFilterFunctionsFullKITTI.py

# Filter the dynamic data according to Mahalanobis distance
python Mahalanobis_dogma.py