import numpy as np


class Particle:
    def __init__(self, fitness_func,
                 dim: int,
                 lower_b: float,
                 upper_b: float,
                 T: float,
                 c1: float,
                 c2: float
                 ):
        self.fitness_func = fitness_func
        self.dim = dim

        self.lb = self.__check_bounds(lower_b)
        self.ub = self.__check_bounds(upper_b)

        self.c1 = c1
        self.c2 = c2
        self.T = T

        self.pos = np.random.uniform(self.lb, self.ub, self.dim)
        self.best_pos = self.pos.copy()

        self.best_fitness_value = self.fitness_value = \
            self.fitness_func(self.pos)

        self.velocity = np.random.uniform(-1, 1, dim)

    def __check_bounds(self, value):
        return value if isinstance(value, list) else [value]*self.dim

    def update_position(self):
        self.pos = np.clip(self.pos + self.velocity, self.lb, self.ub)

        self.fitness_value = self.fitness_func(self.pos) * self.T
        if self.fitness_value < self.best_fitness_value:
            self.best_fitness_value = self.fitness_value
            self.best_pos = self.pos.copy()

    def update_velocity(self, w, global_best_pos):
        r1 = np.random.uniform(0, 1, self.dim)
        r2 = np.random.uniform(0, 1, self.dim)
        inertia = w * self.velocity
        cognitive = self.c1 * r1 * (self.best_pos - self.pos)
        social = self.c2 * r2 * (global_best_pos - self.pos)
        self.velocity = inertia + cognitive + social
