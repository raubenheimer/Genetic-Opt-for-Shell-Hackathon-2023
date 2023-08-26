import pandas as pd
import time
import cvxpy as cp
import numpy as np
from math import ceil
from deap import base, creator, tools, algorithms
from functools import partial
from datetime import datetime
import multiprocessing
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_map(cluster_df, n_clusters):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_df['cluster'] = kmeans.fit_predict(cluster_df[['Lat', 'Long']])
    return cluster_df

def cluster_lob(pred_df,SELECTED_YEAR):
    lob_mass = pred_df[SELECTED_YEAR].sum()*0.2
    cluster_df = pred_df[["Latitude","Longitude",SELECTED_YEAR]]
    grid_points_df = cluster_df[["Latitude","Longitude"]]
    scaler = StandardScaler()
    cluster_df[['Lat', 'Long']] = scaler.fit_transform(cluster_df[['Latitude', 'Longitude']])
    n_clusters = 2
    lob_cluster = 0
    while True:
        cluster_df = cluster_map(cluster_df,n_clusters)
        test_df = cluster_df.groupby('cluster')[SELECTED_YEAR].sum().reset_index()
        test_df = test_df.sort_values(by=SELECTED_YEAR).reset_index()
        #display_clusters(cluster_df, n_clusters)
        if lob_mass>=test_df[SELECTED_YEAR][0]:
            lob_cluster = test_df["cluster"][0]
            #display_clusters(cluster_df,n_clusters)
            break
        n_clusters += 1
    cluster_df = cluster_df[cluster_df['cluster']!=lob_cluster]
    old_to_new_map = cluster_df.reset_index() 
    old_to_new_map = old_to_new_map.rename(columns = {'index' : 'Old'})
    old_to_new_map = old_to_new_map.reset_index()
    old_to_new_map = old_to_new_map.rename(columns = {'index' : 'New'})
    old_to_new_map = old_to_new_map[['Old','New']]
    return old_to_new_map

def gen_depo_pos(length, n_depos):
    """
    Generates a numpy vector of given 'length' with 'n_ones' ones placed at random positions.
    """
    if n_depos > length:
        raise ValueError("Number of ones requested exceeds vector length.")
    vec = np.zeros(length, dtype=bool)
    random_positions = np.random.choice(length, n_depos, replace=False)
    vec[random_positions] = 1
    return vec

def gen_depo_service(length, n_depos):
    """
    Generates a numpy vector of given 'length' with 'n_ones' ones placed at random positions.
    """
    if n_depos > length:
        raise ValueError("Number of ones requested exceeds vector length.")
    vec = np.random.randint(0, n_depos, length)
    matrix = np.zeros((n_depos, length), dtype=bool)
    for col, value in enumerate(vec):
        matrix[value, col] = 1
    return matrix

def create_individual(n_postions,n_depos, toolbox):
    depo_pos = toolbox.depo_pos(length = n_postions,n_depos = n_depos)
    depo_serv_pre = toolbox.depo_serv(length = n_postions,n_depos = n_depos)
    indices = np.where(depo_pos == 1)[0]
    depo_serv = np.zeros((n_postions, n_postions), dtype=bool)
    for i,index in enumerate(indices):
        depo_serv[index] = depo_serv_pre[i]
    individual = np.column_stack((depo_pos, depo_serv))
    # Make the individual an instance of creator.Individual
    individual = creator.Individual(individual)
    return individual

# Fitness function
def evaluate(individual, bio_vec, distance_mat,depo_cap):
    """Objective function."""
    # Extracting depo_pos and depo_serv from individual
    depo_pos = individual[:, 0]
    depo_serv = individual[:, 1:]
    biomass_per_depo = depo_serv @ bio_vec
    penalty_vec = biomass_per_depo[biomass_per_depo>depo_cap]
    penalty_vec = penalty_vec ** 2
    penalty_term = np.sum(penalty_vec) 
    obj = 0.001*np.sum((distance_mat * depo_serv) @ bio_vec) - np.sum(biomass_per_depo) + penalty_term
    obj = float(obj)
    return (obj,)

# Mutation Function
def mutate(ind, n_postions):
    # Randomly select mutation type (1 for depot position, 2 for depot service)
    mutation_type = np.random.randint(1, 15)
    # Random depo pos switch (row)
    if mutation_type == 1:
        switch_1 = np.random.choice(np.where(ind[:, 0] > 0)[0])        
        switch_2 = np.random.randint(0, n_postions)
        ind[[switch_1, switch_2]] = ind[[switch_2, switch_1]]
    
    # Random depo serv switch (column)
    else:
        depo_pos = ind[:, 0]
        depo_serv = ind[:, 1:]
        pos_indices = np.where(depo_pos == 1)
        selected_pos = np.random.choice(pos_indices[0])
        move_service = np.random.randint(0, n_postions)
        depo_serv[:, move_service] = 0
        depo_serv[selected_pos, move_service] = 1
        ind = np.column_stack((depo_pos, depo_serv))
        
    ind = creator.Individual(ind)
    return ind,

# Cross Overfunction
def mate(ind1, ind2, n_postions):
    parent_pos_1 = ind1[:, 0]
    parent_serv_1 = ind1[:, 1:]
    parent_pos_2 = ind2[:, 0]
    parent_serv_2 = ind2[:, 1:]
    #choose child
    parent_pos_idx_1 = np.where(parent_pos_1)
    parent_pos_idx_2 = np.where(parent_pos_2)
    common_depo = np.intersect1d(parent_pos_idx_1, parent_pos_idx_2)
    child_pos_idx_1 = common_depo
    child_pos_idx_2 = common_depo
    pos_choice = np.vstack((np.setdiff1d(parent_pos_idx_1, parent_pos_idx_2), np.setdiff1d(parent_pos_idx_2, parent_pos_idx_1)))
    if len(pos_choice) > 0:
        for j in range(pos_choice.shape[1]):
            switch = np.random.randint(0, 2)
            choice_child_1 = pos_choice[switch,j]
            choice_child_2 = pos_choice[1-switch,j]
            child_pos_idx_1 = np.append(child_pos_idx_1, choice_child_1)
            child_pos_idx_1.sort()
            child_pos_idx_2 = np.append(child_pos_idx_2, choice_child_2)
            child_pos_idx_2.sort()
    child_pos_1 = np.zeros_like(parent_pos_1)
    child_pos_1[child_pos_idx_1] = 1
    child_pos_2 = np.zeros_like(parent_pos_2)
    child_pos_2[child_pos_idx_2] = 1
    combined_serv = np.logical_or(parent_serv_1, parent_serv_2).astype(bool)
    # Build child 1 serv
    child_serv_1 = np.zeros((n_postions,n_postions), dtype=bool)
    for row_idx in child_pos_idx_1:
        child_serv_1[row_idx] = combined_serv[row_idx]
    check_sum_1 = np.sum(child_serv_1, axis=0)
    for col, service_sum in enumerate(check_sum_1):
        if service_sum == 1:
            continue
        elif service_sum == 0:     
            switch_on_choice = np.random.choice(child_pos_idx_1)
            child_serv_1[switch_on_choice,col] = 1
        elif service_sum == 2:
            switch_off_choices = np.where(child_serv_1[:,col])[0]
            switch_off_choice = np.random.choice(switch_off_choices)
            child_serv_1[switch_off_choice,col] = 0
    # Build child 2 serv
    child_serv_2 = np.zeros((n_postions,n_postions), dtype=bool)
    for row_idx in child_pos_idx_2:
        child_serv_2[row_idx] = combined_serv[row_idx]
    check_sum_2 = np.sum(child_serv_2, axis=0)
    for col, service_sum in enumerate(check_sum_2):
        if service_sum == 1:
            continue
        elif service_sum == 0:     
            switch_on_choice = np.random.choice(child_pos_idx_2)
            child_serv_2[switch_on_choice,col] = 1
        elif service_sum == 2:
            switch_off_choices = np.where(child_serv_2[:,col])[0]
            switch_off_choice = np.random.choice(switch_off_choices)
            child_serv_2[switch_off_choice,col] = 0
    # Build children
    child1 = creator.Individual(np.column_stack((child_pos_1, child_serv_1)))
    child2 = creator.Individual(np.column_stack((child_pos_2, child_serv_2)))

    return child1, child2



def optimal_MILP_solve(n_depos, bio_vec, distance_mat,DEPO_CAP):
    n = len(bio_vec)
    # Create the variable depot matrix of 1485x1485 elements
    depo_service = cp.Variable((n, n), nonneg=True)
    
    # Binary decision variable for whether a depot is placed at a location i
    depo_pos = cp.Variable(n, boolean=True)

    # Add the objective function
    obj = cp.Minimize(0.001*cp.sum(cp.multiply(distance_mat, depo_service) @ bio_vec) - cp.sum(depo_service @ bio_vec)) #+ DEPO_CAP*cp.sum(depo_pos)

    # Add the amount of biomass procured per harvesting site = biomass forecated for harvesting site
    # This case is stricter than the problem statement constraint. It has to be because I've already dropped 20% of the biomass. This can be improved in the next iter.
    col_summer = np.ones(n)
    constraints = [depo_service.T @ col_summer == 1]

    # Ensure the allocation to a depot is zero if the depot is not active
    for i in range(n):
        constraints.append(depo_service[i,:] <= depo_pos[i])

    # Add the depot processing capacity constraint
    constraints.append(depo_service @ bio_vec <= 20000)

    # Add the total processing constraint
    constraints.append(cp.sum(depo_service @ bio_vec) >= 0.8*np.sum(bio_vec))

    # Add max depot constraint
    constraints.append(cp.sum(depo_pos) == n_depos)

    # Solve the problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK,verbose=False)
    
    optimal_indv = np.column_stack((depo_pos.value, depo_service.value))
    optimal_fitness = prob.value

    return optimal_indv,optimal_fitness


def main(toolbox):

    #######PARAMETERS#######
    export_dir = "../optim_files"
    SELECTED_YEAR = "2018"
    DEPO_CAP = 20000
    ##Load Opt Data
    dist_df = pd.read_csv("Distance_Matrix.csv")
    dist_df = dist_df.drop(columns="Unnamed: 0")
    predection_values = pd.read_feather("predictions_for_2018_2019.ftr")
    # Do cluster reduction
    old_to_new_map = cluster_lob(predection_values,SELECTED_YEAR)
    old_to_new_map.to_feather(os.path.join(export_dir,"old_to_new_map.ftr"))
    keep_values = old_to_new_map["Old"].values
    # Create Distance Matrix
    distance_mat = dist_df.values
    distance_mat = distance_mat[keep_values][:, keep_values]
    # Create Bio Vec
    bio_vec = predection_values[SELECTED_YEAR].values
    bio_vec = bio_vec[keep_values]
    # Calculate number of depos
    bio_total = np.sum(predection_values[SELECTED_YEAR].values)*0.8
    n_depos = ceil(bio_total/DEPO_CAP) 
    # Set number of locations
    n_postions = len(bio_vec)
    #Population Size
    pop_size = 1000
    #Number Generations
    n_gen = 150
    # Probability of mating
    prob_mating = 0.7
    # Probability of mutating
    prob_mutating = 0.2
    #########################


    # Individual setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    
    # Register the depot service attribute generator
    toolbox.register("depo_serv", gen_depo_service, length=n_postions, n_depos=n_depos)

    # Register the depot position attribute generator
    toolbox.register("depo_pos", gen_depo_pos, length=n_postions, n_depos=n_depos)

    # Individual generator
    create_individual_fixed = partial(create_individual, n_postions=n_postions, n_depos=n_depos, toolbox=toolbox)
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_fixed)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register Fitness function
    toolbox.register("evaluate", partial(evaluate, bio_vec=bio_vec, distance_mat=distance_mat, depo_cap=DEPO_CAP))

    # Register the mustate function
    toolbox.register("mutate", mutate, n_postions=n_postions)
    
    # Register the crossover function
    toolbox.register("mate", mate, n_postions=n_postions)

    # Register the selection function
    toolbox.register("select", tools.selBest)

    # Configure HOF for numpy array
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    pop = toolbox.population(n=pop_size)
    start_fit = map(toolbox.evaluate, pop)
    fitnesses_list = list(start_fit)
    mean_fitness_start = sum(fit[0] for fit in fitnesses_list) / len(fitnesses_list)
    
    # Benchmark Start
    start_time = time.time()
    
    print(f"Starting Optimization: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    final_pop = algorithms.eaMuPlusLambda(pop, toolbox, 300, 700, prob_mating, prob_mutating, ngen=n_gen, stats=None, halloffame=hof, verbose=True)
    #Benchmark End
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.6f} seconds")
    #pool.close()
    best_individual = hof[0]
    #Print Random Fitness
    print(f"Random Fitness: {mean_fitness_start}")
    print(f"Best individual is:\n {best_individual}\nwith fitness: {best_individual.fitness}")
    #Export
    date_string = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    np.save(os.path.join(export_dir,f'depo_pos_ga_opt.npy'), best_individual)


if __name__ == '__main__':
    #pool = multiprocessing.Pool(4)
    # Create toolbox
    toolbox = base.Toolbox()
    #toolbox.register("map", pool.map)
    main(toolbox)