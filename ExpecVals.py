"""
Anne Stratman
Ben Riordan
Mar. 27th, 2018
Computational Lab in Quantum Mechanics

Calculates expectation values for the discrete and harmonic oscillator solvers for the Schrodinger equation
"""

import DiscreteSolver as DS
import HoSolver as HS
import matplotlib.pyplot as plt
import numpy as np

electron_mass = 511

def square_well_potential(x):
    return 0
square_well = (square_well_potential, "Square Well")

def xExpecVal(n_steps):
    positionMatrix = np.zeros((n_steps,n_steps))
    for i in range(0,n_steps):
        for j in range(0,n_steps):
            if i == j:
                positionMatrix[i][j] = 1
            positionMatrix[i][j] = x * positionMatrix[i][j]
    return positionMatrix


DS.run(square_well, -0.3, 0.3, 100, electron_mass, 0, 5, solver = 1)
#plt.show()
DiscreteSolver = DS()
working_eigenvectors = DiscreteSolver.self.eigenvectors
print(working_eigenvectors)

#x = 6
#print(xExpecVal(7))
