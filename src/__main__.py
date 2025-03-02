from .benchmark import Benchmark
from .functions import benches
from .sa_particle_swarm_optimizer import SAParticleSwarmOptimizer

Benchmark(SAParticleSwarmOptimizer).bench(
    benches, N=50, max_iter=1000, visualization=False)
