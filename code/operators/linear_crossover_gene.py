import numpy as np

from pymoo.core.crossover import Crossover


class LXg(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, _, X, **kwargs):
        '''
        This crossover operator linearly combines the two individuals. The factor for how much to include from each parent is generated for each gene separately, similar to the uniform crossover.
        '''
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var))


        _X = np.zeros(X.shape)
        _X[0] = M * X[0] + (1 - M) * X[1]
        _X[1] = (1 - M) * X[0] + M * X[1]

        return _X
