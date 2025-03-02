import numpy as np


class Function:
    def __init__(self, f, dim, bounds, name=None):
        self.f = f
        self.dim = dim
        self.bounds = bounds
        self.name = name if name else f.__name__


def F1(sol):
    return np.sum(np.array(sol)**2)


def F2(sol):
    mods = np.abs(np.array(sol))
    return np.sum(mods) + np.prod(mods)


def F3(sol):
    return np.sum(np.sum(sol[:i+1])**2 for i in range(len(sol)))


def F4(sol):
    return np.max(np.abs(sol))


def F5(sol):
    res = 0
    for i in range(len(sol)-1):
        res += 100*(sol[i+1] - sol[i] **
                    2)**2 + (sol[i] - 1)**2
    return res


def F6(sol):
    return np.sum((np.array(sol)+0.5).astype(np.int_)**2)


def F7(sol):
    return np.sum(i*sol[i]**4 + np.random.rand() for i in range(len(sol)))


def F8(sol):
    return np.sum([-x*np.sin(np.sqrt(np.abs(x))) + len(sol)*418.98288727243369
                   for x in sol])


def F9(sol):
    res = 0
    for i in range(len(sol)):
        x = sol[i]
        res += x**2 - 10*np.cos(2*np.pi*x) + 10
        return res


def F10(sol):
    sum_1 = np.sum([x**2 for x in sol])
    sum_2 = np.sum([np.cos(2*np.pi*x) for x in sol])
    div = 1 / len(sol)
    return -20*np.exp(-0.2*np.sqrt(div*sum_1)) \
        - np.exp(div*sum_2) + 20 + np.e


def F11(sol):
    sum_1 = np.sum([x**2 for x in sol])
    prod = np.prod([np.cos(x/np.sqrt(i+1) + 1)
                    for i, x in enumerate(sol)])
    return 1/4000*sum_1 - prod + 1


benches = [
    Function(F1, 10, (-100, 100)),
    Function(F2, 10, (-10, 10)),
    Function(F3, 10, (-100, 100)),
    Function(F4, 10, (-100, 100)),
    Function(F5, 10, (-30, 30)),
    Function(F6, 10, (-100, 100)),
    Function(F7, 10, (-1.28, 1.28)),
    Function(F8, 10, (-500, 500)),
    Function(F9, 10, (-5.12, 5.12)),
    Function(F10, 10, (-32, 32)),
    Function(F11, 10, (-600, 600)),
]
