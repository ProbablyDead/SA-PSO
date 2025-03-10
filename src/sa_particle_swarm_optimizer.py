import numpy as np
import copy
from .particle import Particle


class SAParticleSwarmOptimizer:
    def __init__(self,
                 fitness_func,
                 lower_b,
                 upper_b,
                 N=30,
                 dim=10,
                 max_iter=1000,
                 w_min=0.4,
                 w_max=0.9,
                 T=0.02,
                 c1=1.5,
                 c2=2.5,
                 visualization=True,
                 stop_value=None):
        self.N = N
        self.dim = dim
        self.max_iter = max_iter
        self.w_min = w_min
        self.w_max = w_max
        self.stop_value = stop_value
        self.visualization = visualization

        self.particle_params = {
            "fitness_func": fitness_func,
            "dim": self.dim,
            "lower_b": lower_b,
            "upper_b": upper_b,
            "T": T,
            "c1": c1,
            "c2": c2
        }

    def __update_inertia_weight(self, best_fitness_value):
        return (3 - np.exp(-self.N / 200) + (best_fitness_value / (8 * self.dim)) ** 2) ** -1 + 0.8

    def optimize(self):
        swarm = [Particle(**self.particle_params) for _ in range(self.N)]

        best_particle = copy.deepcopy(
            min(swarm, key=lambda x: x.fitness_value)
        )

        rng = range(self.max_iter)

        if self.visualization:
            import tqdm
            rng = tqdm.tqdm(rng)

        for _ in rng:
            if self.stop_value and \
                    best_particle.fitness_value < self.stop_value:
                break

            w = np.clip(self.__update_inertia_weight(best_particle.fitness_value),
                        self.w_min,
                        self.w_max)
            for particle in swarm:
                particle.update_velocity(w, best_particle.pos)
                particle.update_position()
                if particle.best_fitness_value < best_particle.fitness_value:
                    best_particle = copy.deepcopy(particle)

        return best_particle.pos, best_particle.fitness_value
