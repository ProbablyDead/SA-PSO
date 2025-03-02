# Self-Adaptive Particle Swarm Optimizer
##### This work solves the **task 04** for the *mark 9*

### Input data
Input data (functions implementations, dimensions and bounds) is in the **`src/functions.py`** file

### Output data
Output data stored in the **`output.txt`** file
Parameters for algorithm benchmark are:
- `N`: population size = 50
- `dim`: number of dimensions = 10
- `max_iter`: maximum iterations count = 1000
- `fitness_function_satisfying_value` = 0

### Algorithm implementation
Realization of position and velocity updating are in **Particle** class, placed in **`src/particle.py`**
Whole algorithm (finding the best solution) is implemented in **`src/sa_particle_swarm_optimizer.py`**, under the **SAParticleSwarmOptimizer** class

### How to run this code
- Download dependencies:
```sh
pip install -r requirements.txt
```
- Run:
```sh
python -m src
```
