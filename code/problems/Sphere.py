import numpy as np
import pymoo.gradient.toolbox as anp
from pymoo.problems.single.sphere import Sphere


class Sphere(Sphere):
    def __init__(self, n_var=10, xl=-5, xu=5):
        super().__init__(n_var=n_var)
        self.xl=np.ones(n_var)*xl
        self.xu=np.ones(n_var)*xu
    
    def _evaluate(self, x, out, *args, **kwargs):#we have to overwrite this her
        out["F"] = anp.sum(anp.square(x), axis=1)