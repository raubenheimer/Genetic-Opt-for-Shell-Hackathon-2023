import pandas as pd
import time
import cvxpy as cp
import numpy as np
from deap import base, creator, tools, algorithms
from functools import partial
from datetime import datetime
from math import ceil
import os
import multiprocessing

def create_individual(depo_pos,n_postions,n_ref):
    if n_ref > n_postions:
        raise ValueError("Number of ones requested exceeds vector length.")
    #generate ref positions
    ref_pos = np.zeros(n_postions, dtype=bool)
    ref_pos_idx = np.random.choice(n_postions, n_ref, replace=False)
    ref_pos[ref_pos_idx] = 1
    #generate ref service
    depo_pos_idx = np.where(depo_pos)[0]
    ref_serv = np.zeros((n_postions,n_postions),dtype=bool)
    for i in range(len(depo_pos_idx)):
        random_ref_pos = np.random.choice(ref_pos_idx)
        ref_serv[random_ref_pos,depo_pos_idx[i]] = 1
    individual = np.column_stack((ref_pos, ref_serv)).astype(bool)
    # Make the individual an instance of creator.Individual
    individual = creator.Individual(individual) 
    return individual

# Fitness function
def evaluate(individual, depo_vec, distance_mat):
    """Objective function."""
    # Extracting depo_pos and depo_serv from individual
    ref_pos = individual[:, 0]
    ref_serv = individual[:, 1:]
    obj = 0.001*np.sum((distance_mat * ref_serv) @ depo_vec) - np.sum(ref_serv @ depo_vec)
    obj = float(obj)
    return (obj,)

# Mutation Function
def mutate(ind, n_postions, depo_pos):
    # Randomly select mutation type (1 for depot position, 2 for depot service)
    mutation_type = np.random.randint(1, 15)
    # Random depo pos switch (row)
    if mutation_type == 1:
        switch_1 = np.random.choice(np.where(ind[:, 0] > 0)[0])        
        switch_2 = np.random.randint(0, n_postions)
        ind[[switch_1, switch_2]] = ind[[switch_2, switch_1]]
    
    # Random depo serv switch (column)
    else:
        ref_pos = ind[:, 0]
        ref_serv = ind[:, 1:]
        pos_indices = np.where(ref_pos == 1)[0]
        selected_pos = np.random.choice(pos_indices)
        move_service = np.random.choice(np.where(depo_pos == 1)[0])
        ref_serv[:, move_service] = 0
        ref_serv[selected_pos, move_service] = 1
        ind = np.column_stack((ref_pos, ref_serv))
    ind = creator.Individual(ind)
    return ind,

# Cross Overfunction
def mate(ind1, ind2, n_postions, depo_pos):
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
        if depo_pos[col] == 0:
            continue
        elif service_sum == 1:
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
        if depo_pos[col] == 0:
            continue
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

def optimal_lp_solve(opt_indv, depo_pos, depo_vec, distance_mat,REF_CAP):
    n = len(depo_vec)

    # Refinery positions from GA
    ref_pos = opt_indv[:, 0]

    # Create the variable depot matrix of 1485x1485 elements
    ref_service = cp.Variable((n, n), nonneg=True)
    ref_service.value = np.array(opt_indv[:, 1:])
    
    # Add the objective function
    obj = cp.Minimize(0.001*cp.sum(cp.multiply(distance_mat, ref_service) @ depo_vec) - cp.sum(ref_service @ depo_vec)) #+ DEPO_CAP*cp.sum(depo_pos)

    # Add the refinery processing capacity constraint
    constraints = [ref_service @ depo_vec <= REF_CAP]

    # Ensure the allocation to a refinery is zero if there is no refinery in that position
    for j in range(n):
        constraints.append(cp.sum(ref_service[:,j]) <= depo_pos[j])

    # Ensure the allocation to a refinery is zero if the refinery is not active
    for i in range(n):
        constraints.append(ref_service[i,:] <= ref_pos[i])

    # Add the total processing constraint
    constraints.append(cp.sum(ref_service @ depo_vec) == np.sum(depo_vec))

    # Solve the problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK,verbose=True,warm_start=True)
    
    optimal_indv = np.column_stack((ref_pos, ref_service.value))
    optimal_fitness = prob.value

    return optimal_indv,optimal_fitness

def main(toolbox):






    #######PROGRAM FLOW#######
    run_ref_ga = True
    #######PARAMETERS#######
    export_dir = "../optim_files"
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
    # Calculate number of depos
    DEPO_CAP = 20000
    REF_CAP = 100000
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
    # Number of refs to place
    n_ref = ceil(bio_total/REF_CAP) 
    # Depo positions
    depo_pos_file = "depo_positions.npy"
    depo_serv_file = "mosek_depo_serv_optim.npy"
    depo_pos = np.load(os.path.join(export_dir,depo_pos_file)).astype(int)
    depo_serv = np.load(os.path.join(export_dir,depo_serv_file))
    depo_serv_zero_idx = np.where(depo_serv <= 0.00001)
    depo_serv[depo_serv_zero_idx] = 0
    depo_vec = depo_serv @bio_vec
    #########################

    #optimal_indv,optimal_fitness = optimal_solve(n_depos, bio_vec, distance_mat,DEPO_CAP)
    # Individual setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    # Individual generator
    create_individual_fixed = partial(create_individual,depo_pos=depo_pos, n_postions=n_postions, n_ref=n_ref)
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_fixed)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register Fitness function
    toolbox.register("evaluate", partial(evaluate, depo_vec=depo_vec, distance_mat=distance_mat))

    # Register the mustate function
    toolbox.register("mutate", mutate, n_postions=n_postions, depo_pos=depo_pos)
    
    # Register the crossover function
    toolbox.register("mate", mate, n_postions=n_postions, depo_pos=depo_pos)

    # Register the selection function
    toolbox.register("select", tools.selBest)

    # Configure HOF for numpy array
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    pop = toolbox.population(n=pop_size)
    start_fit = map(toolbox.evaluate, pop)
    fitnesses_list = list(start_fit)
    mean_fitness_start = sum(fit[0] for fit in fitnesses_list) / len(fitnesses_list)
    
    #optimal_indv,optimal_fitness = optimal_solve(n_ref, depo_pos, depo_vec, distance_mat,REF_CAP)
    # zero_idx = np.where(optimal_indv <= 0.00001)
    # optimal_indv[zero_idx] = 0
    # optimal_indv = optimal_indv.astype(int)

    ##GA OPTIMIZE##
    # Benchmark Start
    if run_ref_ga:
        start_time = time.time()
        print("Starting Genetic Opt:")
        final_pop = algorithms.eaMuPlusLambda(pop, toolbox, 300, 700, prob_mating, prob_mutating, ngen=n_gen, stats=None, halloffame=hof, verbose=True)
        #pool.close()
        best_individual = hof[0]
        date_string = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
        #Save result
        np.save(os.path.join(export_dir,f'ref_hof_indv-{date_string}.npy'), best_individual)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed GA Time: {elapsed_time:.6f} seconds")
    else:
        print("Loading Optimized Individual:")
        best_individual = np.load('../optimized/set0/ref_hof_indv.npy')

    ##LP OPTIMIZE##
    start_time = time.time()
    print("Starting LP Opt:")
    best_individual, best_fitness = optimal_lp_solve(best_individual, depo_pos, depo_vec, distance_mat,REF_CAP)
    date_string = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    np.save(os.path.join(export_dir,f'ref_lp_hof_indv.npy'), best_individual)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed LP Time: {elapsed_time:.6f} seconds")

if __name__ == '__main__':
    #pool = multiprocessing.Pool(4)
    # Create toolbox
    toolbox = base.Toolbox()
    #toolbox.register("map", pool.map)
    main(toolbox)