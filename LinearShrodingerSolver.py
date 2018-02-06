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
import matplotlib.pyplot as plt
        
class Solver:
"""
"""
    def __init__(self, potential, xmin, xmax, n_steps):
        """
        Variables:
        self (obj)
        potential(function)
        xmin(float)
        xmax(float)
        n_steps(int)
        """
        self.potential = potential
        self.xmin = xmin
        self.xmax = xmax
        self.n_steps = n_steps
        
        self.h = (self.xmax-self.xmin)/self.n_steps
        self.xPoints = np.zeroes(0,n_steps-1)
        for i in range(0, n_steps-1):
            xPoints[i] = i*self.h + self.xmin

        
    def matrix_element_finder(self,i,j): 
        """
        Calculates the i-jth element of the matrix
        All elements are nonzero except diagonal and off-diagonal elements
        """

        if i == j:
            #Potential is evaluated at discrete points
            Element = 2/(h**2) + self.potential(xPoints[i])
        elif i == j + 1:
            Element = -1/(h**2)
        elif j == i + 1:
            Element = -1/(h**2)
        else:
            Element = 0
        return Element

    def matrix_maker(self):
    """
    Creates a matrix and stores the values of the matrix found by Solver.matrix_element_finder as the elements of the matrix.
    
    Returns:
    a (numpy array) - The matrix with elements formed by matrix_element_finder
    """
        self.a = np.zeros((self.n_steps,self.n_steps))
        for i in range(0, self.n_steps-1):
            for j in range(0, self.n_steps-1):
                a[i][j] = matrix_element_finder(i,j)
        

    def matrix_solver(self):
    """
    Diagonalizes the matrix

    Returns:
    numpy array of eigenvalues - energies
    numpy array of eigenvectors - values of wavefunction corresponding to each energy
    """
        self.eigenvalues, self.eigenvectors = np.linalg.eig(a)








