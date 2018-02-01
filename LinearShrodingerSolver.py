"""Solves Schrodinger equation

Anne Stratman
Ben Riordan
Jan. 25th, 2018
Computational Lab in Quantum Mechanics

Steps:
1. Transform radial Schrodinger equation to a dimensionless? form
2. Rewrite as a matrix eigenvalue problem
	a. Set up array of x-values
	b. Set up equation for potential and evaluate potential at x-values to generate array of potential values
	c. Construct 1D array of diagonal matrix elements
	d. Construct 1D array of off-diagonal matrix elements
	e. Construct matrix
	f. Use numpy eigensolver to diagonalize matrix
3. Results: eigenvectors are states, eigenvalues are energies
4. Write output file with energies and eigenvectors as rows

Tests:
1. Infinite square well


"""
import numpy as np

"""def nlsSolver(potential, xmin, xmax, n_steps):

   note: n_steps = basis dimension + 1 


        ""
"""
        
class Solver:
"""
"""
    def __init__(self, potential, xmin, xmax, n_steps):
        self.potential = potential
        self.xmin = xmin
        self.xmax = xmax
        self.n_steps = n_steps
        
    def matrix_element_finder(i,j):
    """
    """
        h = (xmax-xmin)/n_steps
        for i in range(1, n_steps-1):
            ""
            
    def matrix_maker():
    """
    """
        ""
        
    def matrix_solver():
    """
    """
        ""

"""Define variables"""
