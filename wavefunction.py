
import numpy as np
from typing import List, Tuple
import math


spin_choices = [1, 0]

class Wavefunction():

    def __init__(self, n_electrons):

        self.n_electrons = n_electrons
        self.positions = [np.random.rand(3) for i in range(n_electrons)]
        
    def get_positions(self) -> List[List[float]]:
        return self.positions
    
    def update_positions(self, new_positions: List[List[float]]):
        assert len(self.positions) ==len(new_positions)
        self.positions = new_positions


class TrialExp(Wavefunction):

    def __init__(self, n_electrons):
        super.__init__(self, n_electrons)

    def evaluate(self, coords, alpha):
        return math.exp(-alpha * (modulo(coords[0]) ** 2 + modulo(coords[1]) ** 2))
        