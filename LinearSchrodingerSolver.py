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

Units:
hbar = 1
c = 1

Neutron mass mN*c**2: 938 MeV
hbar*c = 197 MeV fm
Units of hbar*c / mN*c**2: MeV fm**2

"""
import numpy as np
import matplotlib.pyplot as plt
        
class Solver:
    def __init__(self, potential, xmin, xmax, n_steps, particle_mass):
        """
        Arguments:
        self (obj)
        potential(function) - potential function to use in solving the NLS
        xmin(float) - left bound of position
        xmax(float) - right bound of position
        n_steps(int) - -number of increments in interval
        particle_mass (float) - mass of particle in keV
        """
        self.potential = potential
        self.xmin = xmin
        self.xmax = xmax
        self.n_steps = n_steps
        self.mass = particle_mass

        
        self.h = (self.xmax-self.xmin)/self.n_steps
        
        self.xPoints = np.zeros(n_steps+1)
        for i in range(n_steps+1):
            self.xPoints[i] = i*self.h + self.xmin
        
    def matrix_element_finder(self,i,j): 
        """
        Calculates the i-jth element of the matrix
        All elements are nonzero except diagonal and off-diagonal elements
        
        Arguments:
        i (int) - the row of the matrix to calculate
        j (int) - the column of the matrix to calculate
        
        Returns: 
        Element (float) - calculated ij-th element of matrix
        """
        if i == j:
            #Potential is evaluated at discrete points
            #hbar = 1
            #Multiply each term by 1/(2*m)
            Element = 2/((self.h**2)*2*self.mass) + self.potential(self.xPoints[i])
        elif i == j + 1:
            Element = -1/((self.h**2)*2*self.mass)
        elif j == i + 1:
            Element = -1/((self.h**2)*2*self.mass)
        else:
            Element = 0
        return Element

    def matrix_maker(self):
        """
        Creates a matrix and stores the values of the matrix found by Solver.matrix_element_finder as the elements of the matrix.
    
        Returns:
        a (numpy array) - The matrix with elements formed by matrix_element_finder
        """
        self.a = np.zeros((self.n_steps+1,self.n_steps+1))
        for i in range(1, self.n_steps):
            for j in range(1, self.n_steps):
                self.a[i][j] = self.matrix_element_finder(i,j)
        

    def matrix_solver(self):
        """
        Diagonalizes the matrix
    
        Returns:
        numpy array of eigenvalues - energies
        numpy array of eigenvectors - values of wavefunction corresponding to each energy
        """
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.a)
        self.eigenvectors = np.transpose(self.eigenvectors)
        
        #Normalization of the eigenvectors
        for i in range(0, len(self.eigenvectors)):
            self.eigenvectors[i] = (1/np.sqrt(self.h)) * self.eigenvectors[i]
        
def nrg_plot(psi, n, m = None):
    """
    Plots the eigenvectors and eigenvalues for a certain hamiltonian over a range of n values or at a single n value.
    
    Arguments:
    psi (Solver obj) - an object representing a specific hamiltonian
    n (int) - lower bound of eigenvectors to plot
    m (int) [OPTIONAL] - upper bound of eigenvectors to plot
    """
    if m == None:
        plt.plot(psi.xPoints,psi.eigenvectors[n+1])
    else:
        for i in range(n,m):
            plt.plot(psi.xPoints,psi.eigenvectors[i+1])

    plt.ylabel('WaveFunction')
    plt.xlabel('Position')
    plt.show()

def run(p_function, xmin, xmax, dim, n, m = None, x_points = None, e_values = None, e_vectors = None, hamiltonian = None):
    """
    Creates a solver object for a potential function and plots the potential function's wavefunction.
    
    Arguments:
    p_function (function) - a potential function
    xmin (float) - left bound of positions
    xmax (float) - right bound of positions
    dim (int) - number of increments when evaluating the wavefunctions
    n (int) - lower bound of eigenvectors to plot
    m (int) [OPTIONAL] - upper bound of eigenvectors to plot
    x_points (bool) [OPTIONAL] - if True, prints the xPoints array
    
    """
    potential = Solver(p_function, xmin, xmax, dim)
    
    potential.matrix_maker()
    potential.matrix_solver()
    
    if x_points != None:
        print(potential.xPoints)
    
    if e_values != None:
        print(potential.eigenvalues)
        
    if e_vectors != None:
        print(potential.eigenvectors)
        
    if hamiltonian != None:
        print(potential.a)
        
    if m == None:
        nrg_plot(potential, n)
    else:
        nrg_plot(potential, n, m)
   
   
if (__name__ == "__main__"):

    def square_well_potential(x):
        return 0

    def ho_potential(x):
        return -(1/2)*x**2
     
    run(square_well_potential, 0, 1, 100, 1, e_vectors = True)
    #squareWell = Solver(square_well_potential, 0, 1, 100)
    #hOscillator = Solver(ho_potential, 0, 1, 100)
    
    #squareWell.matrix_maker()
    #squareWell.matrix_solver()
    
    #hOscillator.matrix_maker()
    #hOscillator.matrix_solver()
    
    #print(squareWell.xPoints)
    #print(hOscillator.xPoints)
    
    #print(squareWell.eigenvalues)
    #print(hOscillator.eigenvalues)
    
    #print(squareWell.eigenvectors)
    #print(hOscillator.eigenvectors)

    #print(squareWell.a)
    #print(hOscillator.a)

    #nrg_plot(squareWell, 1, 10)
    #nrg_plot(hOscillator, 1)
