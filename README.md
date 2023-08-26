# Genetic-Opt-for-Shell-Hackathon-2023
This repository contains the code I used for positioning depots and refineries for the Shell Hackathon 2023. The entire codebase can run on a standard PC. I apologize for any code disarray; time constraints didn't allow me to refine it further.

## Execution Sequence:
To use the scripts in the correct order, follow the sequence below:
1. 'depo_pos_ga_opt.py'
2. 'depo_serv_lp_opt.py'
3. 'ref_pos_ga_opt.py'
4. 'generate_submission.py'

## Script Descriptions:
* depo_pos_ga_opt.py:
    * Trims the largest blob from the map, ensuring processed biomass remains ≥ 80% of the total biomass.
    * Optimizes depot positions using a genetic algorithm. This task is an integer programming problem.
* depo_serv_lp_opt.py:
    * Determines the optimal match between source destinations and their servicing depots. This task is a linear programming problem.
* ref_pos_ga_opt.py:
    * Optimizes refinery positions using a genetic algorithm (integer programming problem).
    * Determines which refineries will service which depots (a linear programming problem).
* generate_submission.py:
    * Produces the submission file.

## Suggestions for imporvement:
The original objective function optimization is a nonlinear mixed integer problem. My approach of positioning the depots prior to the refineries linearizes the problem but may not yield the optimal solution. A more direct approach would involve integrating the two individual genetic algorithms to tackle the nonlinear case. However, due to time constraints, I couldn't explore this avenue. It's worth noting that the search space for such a direct optimization would be vast.

Additionally, I set the number of depots and refineries to the bare minimum required to process 80% of the total biomass. Incorporating these values directly into the genetic algorithm might be a possible enhancement.

Just run increase the population size or number of generations.

## Notes on performance:
Both genetic algorithms were executed for 150 generations with a population size of 1000 individuals. Here are the obtained scores:
* a x Cost_transport = 44702
* c x Cost_underutil = 80000
Unfortunately, my forecasting cost was suboptimal.

