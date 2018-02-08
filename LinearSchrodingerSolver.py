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
        self.xPoints = np.zeros(n_steps+1)
        for i in range(n_steps+1):
            self.xPoints[i] = i*self.h + self.xmin

        
    def matrix_element_finder(self,i,j): 
        """
        Calculates the i-jth element of the matrix
        All elements are nonzero except diagonal and off-diagonal elements
        
        Returns: Element (float) - calculated ij-th element of matrix
        """

        if i == j:
            #Potential is evaluated at discrete points
            Element = 2/(self.h**2) + self.potential(self.xPoints[i-1])
        elif i == j + 1:
            Element = -1/(self.h**2)
        elif j == i + 1:
            Element = -1/(self.h**2)
        else:
            Element = 0
        return Element

    def matrix_maker(self):
        """
    Creates a matrix and stores the values of the matrix found by Solver.matrix_element_finder as the elements of the matrix.

    Need to leave out first and last points and pad matrix with zeros
    
    Returns:
    a (numpy array) - The matrix with elements formed by matrix_element_finder
        """
        self.a = np.zeros((self.n_steps+1,self.n_steps+1))
        for i in range(1, self.n_steps+1):
            for j in range(1, self.n_steps+1):
                self.a[i][j] = self.matrix_element_finder(i,j)
        

    def matrix_solver(self):
        """
    Diagonalizes the matrix

    Returns:
    numpy array of eigenvalues - energies
    numpy array of eigenvectors - values of wavefunction corresponding to each energy
        """
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.a)

def nrg_plot(psi):
    """
    Plots the eigenvectors and eigenvalues for a certain hamiltonian.
    
    Variables:
    potential (Solver obj) - an object representing a specific hamiltonian
    """
    
    #for i in range(0,len(e_vectors)):
            #plt.plot(potential.xPoints,e_vectors[i])
            
    plt.plot(psi.xPoints,psi.eigenvectors[1])
    
    plt.ylabel('WaveFunction')
    plt.xlabel('Position')
    plt.show()


        
if (__name__ == "__main__"):

    def square_well_potential(x):
        return 0

    def ho_potential(x):
        return x*x
        
    squareWell = Solver(square_well_potential, 0,100,1000)
    hOscillator = Solver(ho_potential,0,100,1000)
    
    squareWell.matrix_maker()
    squareWell.matrix_solver()
    
    hOscillator.matrix_maker()
    hOscillator.matrix_solver()
    
    #print(squareWell.xPoints)
    #print(squareWell.eigenvalues)
    #print(squareWell.eigenvectors)

    print('buffer line')
    
    print(squareWell.a)
    print(hOscillator.a)
    #print(squareWell.eigenvectors[0][0])
    nrg_plot(squareWell)
    nrg_plot(hOscillator)
    
    





