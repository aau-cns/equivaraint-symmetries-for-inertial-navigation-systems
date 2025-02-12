# Automatica submission: *Equivaraint Symmetries for Inertial Navigation Systems*
> *Alessandro Fornasier, Yixiao Ge, Pieter van Goor, Robert Mahony, and Stephan Weiss*
---

## Dependencies
* pylie: https://github.com/AlessandroFornasier/pylie
  * Note that the newest version of pylie: https://github.com/pvangoor/pylie is not yet supported

## How to reproduce the results of the submission:

#### Run the filters:
```bash
# To run the filters on the simulated data
python3 Simulation/Simulation.py Trajectories/sim <path_where_to_save_the_results> --ct --ctex --ctnew --iekf --tfiekf --mekf --equivariant_output

# To run the filters on the real data
python3 Simulation/Simulation.py Trajectories/insane <path_where_to_save_the_results> --ct --ctex --ctnew --iekf --tfiekf --mekf --equivariant_output --insane
```

#### Plot the results:
Run matlab's script `Evaluation/scripts/plotFilters.m` to plot the results

