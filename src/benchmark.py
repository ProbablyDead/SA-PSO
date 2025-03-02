class Benchmark:
    def __init__(self, optimizer):
        self.__optimizer = optimizer

    def bench(self, benches, N=50, max_iter=1000, visualization=True):
        for bench in benches:
            res, res_a = self.__optimizer(bench.f,
                                          N=N,
                                          max_iter=max_iter,
                                          dim=bench.dim,
                                          lower_b=bench.bounds[0],
                                          upper_b=bench.bounds[1],
                                          visualization=visualization
                                          ).optimize()
            print(f"{bench.name}\t{res_a}\n{res}\n")
