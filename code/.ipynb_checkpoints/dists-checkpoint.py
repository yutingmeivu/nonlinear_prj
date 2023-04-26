import numpy as np
from numpy.random import Generator, PCG64

def uniform(low, up):
    return np.random.uniform(low, up)
    
def norm(mean, std):
    return np.random.normal(mean, std)

def gen(low, up):
    rng = Generator(PCG64())
    return rng.standard_normal()