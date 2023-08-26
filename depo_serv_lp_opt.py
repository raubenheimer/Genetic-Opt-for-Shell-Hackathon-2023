import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import datetime
import os

export_dir = "../optim_files"
file_to_load = "depo_pos_ga_opt.npy.npy"
indv = np.load(os.path.join(export_dir,file_to_load))

SELECTED_YEAR = "2018"
dist_df = pd.read_csv("Distance_Matrix.csv")
dist_df = dist_df.drop(columns="Unnamed: 0")
predection_values = pd.read_feather("predictions_for_2018_2019.ftr")
# Import Keep Values
keep_values = pd.read_feather(os.path.join(export_dir,"old_to_new_map.ftr"))
keep_values = keep_values["Old"].values
# Create Distance Matrix
distance_mat = dist_df.values
distance_mat = distance_mat[keep_values][:, keep_values]
# Create Bio Vec
bio_vec = predection_values[SELECTED_YEAR].values
bio_vec = bio_vec[keep_values]
n= len(bio_vec)
DEPO_CAP =20000
depo_pos = indv[:, 0]
depo_serv = indv[:, 1:]

depo_indices = np.where(depo_pos == 1)[0]
service_per_depo = depo_serv@bio_vec


# Create the variable depot matrix of 1485x1485 elements
depo_service = cp.Variable((n, n), nonneg=True)
depo_service.value = depo_serv

# Add the objective function
obj = cp.Minimize(0.001*cp.sum(cp.multiply(distance_mat, depo_service) @ bio_vec)  - cp.sum(depo_service @ bio_vec)) #+ DEPO_CAP*cp.sum(depo_pos)

# Add the amount of biomass procured per harvesting site = biomass forecated for harvesting site
# This case is stricter than the problem statement constraint. It has to be because I've already dropped 20% of the biomass. This can be improved in the next iter.
col_summer = np.ones(n)
constraints = [depo_service.T @ col_summer <= 1]

# Ensure the allocation to a depot is zero if the depot is not active
for i in range(n):
    constraints.append(depo_service[i,:] <= depo_pos[i])

# Add the depot processing capacity constraint
constraints.append(depo_service @ bio_vec <= DEPO_CAP)

constraints.append(cp.sum(depo_service @ bio_vec) >= 0.8*cp.sum(bio_vec))

# Solve the problem
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.MOSEK,verbose=True,warm_start =True)

date_string = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
np.save(os.path.join(export_dir,f'mosek_depo_serv_optim.npy'), depo_service.value)
np.save(os.path.join(export_dir,f'depo_positions.npy'), depo_pos)