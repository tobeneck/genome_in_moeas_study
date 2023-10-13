import numpy as np

from pymoo.core.crossover import Crossover


class LXi(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, _, X, **kwargs):
        '''
        This crossover operator linearly combines the two individuals. There is only one factor generated for each pair of individuals.
        '''
        _, n_matings, _ = X.shape
        M = np.random.random((n_matings))

        _X = np.zeros(X.shape)
        _X[0] = (M * X[0].T).T + ((1 - M) * X[1].T).T
        _X[1] = ((1 - M) * X[0].T).T + (M * X[1].T).T

        return _X
