import pandas as pd
import numpy as np

def round_down(arr, decimals=4):
    factor = 10**decimals
    return np.floor(arr * factor) / factor

# Import data
depo_pos_red = np.load('../optim_files/depo_positions.npy').astype(int)
depo_serv_red = np.load('../optim_files/mosek_depo_serv_optim.npy')
ref_indv = np.load('../optim_files/ref_hof_indv.npy')
predection_values = pd.read_feather("predictions_for_2018_2019.ftr")
idx_map = pd.read_feather("../optim_files/old_to_new_map.ftr")
idx_map_dict = idx_map.set_index('New')['Old'].to_dict()
ref_pos_red = ref_indv[:, 0].astype(int)
ref_serv_red = ref_indv[:, 1:]

# Get predection vector
SELECTED_YEAR = '2018'
bio_vec = predection_values[SELECTED_YEAR].values
n = len(bio_vec)


# Reconstruct pos
depo_pos = np.zeros(n)
depo_pos[idx_map["Old"]] = depo_pos_red
ref_pos = np.zeros(n)
ref_pos[idx_map["Old"]] = ref_pos_red
# Reconstruct depo serv
depo_serv = np.zeros((n,n))
for i in range(depo_serv_red.shape[0]):
    mapped_i = idx_map_dict[i]
    for j in range(depo_serv_red.shape[1]):
        mapped_j = idx_map_dict[j]
        if depo_serv_red[i,j] > 0.000000001:
            depo_serv[mapped_i,mapped_j] = depo_serv_red[i,j]
depo_serv = round_down(depo_serv,6)

# Reconstruct ref serv    
ref_serv = np.zeros((n,n))
for i in range(ref_serv_red.shape[0]):
    mapped_i = idx_map_dict[i]
    for j in range(ref_serv_red.shape[1]):
        mapped_j = idx_map_dict[j]  
        ref_serv[mapped_i,mapped_j] = ref_serv_red[i,j]

# first part: Depo and refinery locations
depo_idx = np.where(depo_pos>0)[0]
ref_idx = np.where(ref_pos>0)[0]
df = pd.DataFrame(depo_idx,columns=["source_index"])
df['data_type'] = "depot_location"
df_cat = pd.DataFrame(ref_idx,columns=["source_index"])
df_cat['data_type'] = "refinery_location"
df = pd.concat([df,df_cat])
df['year'] = '20182019'
df = df[['year','data_type','source_index']]
# second part: biomass forecast
df_cat = pd.DataFrame(bio_vec,columns=["value"])
df_cat['year'] = '2018'
df_cat['data_type'] = 'biomass_forecast'
df_cat = df_cat.reset_index()
df_cat = df_cat.rename(columns={'index':'source_index'})
df_cat = df_cat[['year','data_type','source_index','value']]
df = pd.concat([df,df_cat])
df['value'] = df['value'].fillna('')
# third part: depo service per position
depo_service_flat = []
bio_mat_multip = np.tile(bio_vec, (len(bio_vec), 1))
depo_service_absolute = depo_serv * bio_mat_multip
#depo_service_absolute = round_down(depo_service_absolute,5)
for i in range(depo_service_absolute.shape[0]):
    for j in range(depo_service_absolute.shape[1]):
        if depo_service_absolute[i, j] >= 0.0000000001:
            depo_service_flat.append([i,j,depo_service_absolute[i, j]])
df_cat = pd.DataFrame(depo_service_flat,columns=["destination_index","source_index","value"])
df_cat['year'] = '2018'
df_cat['data_type'] = "biomass_demand_supply"
df_cat = df_cat[["year","data_type","source_index","destination_index","value"]]
df = pd.concat([df,df_cat])
df['destination_index'] = df['destination_index'].fillna('')
test = depo_serv @ bio_vec
# forth part: ref service per depo
ref_service_flat = []
depo_vec = depo_serv @bio_vec
depo_vec_multip = depo_vec#.reshape(-1, 1)
ref_service_absolute = ref_serv * depo_vec_multip
for i in range(ref_service_absolute.shape[0]):
    for j in range(ref_service_absolute.shape[1]):
        if ref_service_absolute[i, j] >= 0.01:
            ref_service_flat.append([i,j,ref_service_absolute[i, j]])
df_cat = pd.DataFrame(ref_service_flat,columns=["destination_index","source_index","value"])
df_cat['year'] = '2018'
df_cat['data_type'] = "pellet_demand_supply"
df_cat = df_cat[["year","data_type","source_index","destination_index","value"]]
df = pd.concat([df,df_cat])
# Final part: duplication of other year
other_year_df = df[df['year'] == SELECTED_YEAR]
other_year = "2019"
if SELECTED_YEAR == "2019":
    other_year = "2018"
other_year_df['year'] = other_year
df = pd.concat([df,other_year_df])
df = df[["year","data_type","source_index","destination_index","value"]]
df.to_csv('submission_1.csv', index=False)