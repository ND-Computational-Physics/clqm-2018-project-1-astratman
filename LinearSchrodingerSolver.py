"""
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
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.misc
import scipy.special
          
class Discrete_Solver:
    def __init__(self, potential, xmin, xmax, n_steps, particle_mass):
        """
        Arguments:
        self (obj)
        potential(function) - potential function to use in solving the NLS
        xmin(float) - left bound of position
        xmax(float) - right bound of position
        n_steps(int) - -number of increments in interval
        particle_mass (float) - mass of particle in keV
        hbarc is in MeV fm
        """
        self.potential = potential
        self.xmin = xmin
        self.xmax = xmax
        self.n_steps = n_steps
        self.mass = particle_mass

        self.h = (self.xmax-self.xmin)/self.n_steps
        
        self.xPoints = np.zeros(self.n_steps+1)
        for i in range(self.n_steps+1):
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
            #hbar*c = 197
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
        """
        self.a = np.zeros((self.n_steps+1,self.n_steps+1))
        for i in range(1, self.n_steps):
            for j in range(1, self.n_steps):
                self.a[i][j] = self.matrix_element_finder(i,j)

    def matrix_solver(self):
        """
        Finds a matrix's eigenvalues and (normalized) eigenvectors
        """
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.a)
        self.eigenvectors = np.transpose(self.eigenvectors)
        
        #Normalization of the eigenvectors
        for i in range(0, len(self.eigenvectors)):
            self.eigenvectors[i] = (1/np.sqrt(self.h)) * self.eigenvectors[i]
        
class Ho_Solver:
    def __init__(self, potential, xmin, xmax, n_functions, particle_mass, omega):
        """
        Attributes:
        \Initialized\
        self (obj)
        potential(function) - potential function to use in solving the NLS
        xmin(float) - left bound of position
        xmax(float) - right bound of position
        n_functions(int) - -number of eigenfunctions to find
        particle_mass (float) - mass of particle in keV
        omega (float) - frequency of HO
        
        \Assigned\
        n_steps(int) - number of steps to take within the x range
        h(float) - the spacing between each x point
        xPoints(float array) - a 1D array of x points
        transform(float array) - a 2D array used to change basis from the HO basis to the discrete basis
        hamiltonian(float array) - the hamiltonian operator matrix (2D array)
        eigenvalues(float array) - a 1D array of the eigenvalues for our potential
        eigenvectors(float array) - a 2D array of eigenevectors for our potential
        """
        self.potential = potential
        self.xmin = xmin
        self.xmax = xmax
        self.n_steps = 100 #want to be rows
        self.n_functions = n_functions #want to be columns
        self.mass = particle_mass
        self.omega = omega

        self.h = (self.xmax-self.xmin)/self.n_steps
        
        self.xPoints = np.zeros(self.n_steps)
        for i in range(self.n_steps):
            self.xPoints[i] = i*self.h + self.xmin

    def HO_wavefunction(self,x,n):
        """
        Defines the harmonic oscillator wavefunction
        hermval evaluates a Hermite series, so subtracts the value of the (n-1)th series from the value
            of the nth series to get the value of the nth Hermite polynomial
        Arguments:
            x (float): x coordinate to evaluate wavefunction at
            n (int): index of hermite polynomial and of wavefunction
        Returns:
            value of wavefunction (float)

        """
        psi = self.mass**(1/4) * self.omega**2 * (1/(np.sqrt(float(2**n))*scipy.misc.factorial(n))) * scipy.special.hermite(n) * np.exp(-x**2/2)
        
        Psi = 0
        for i in (range(len(psi))):
            Psi += psi[i]*x**i
        return Psi

    def HO_matrix(self):
        """
        Creates a matrix with different harmonic oscillator wavefunctions evaluated at a vector of xPoints
        
        Arguments:
        n_steps(int) - number of rows, each row corresponds to different wavefunctions evaluated at the same point
        n_functions(int) - number of columns, each column corresponds to the same wavefunction evaluated at different points
        """
        self.transform = np.zeros((self.n_steps,self.n_functions))
        for i in range(len(self.xPoints)):
            x = self.xPoints[i]
            for j in range(0,self.n_functions):
                self.transform[i][j] = self.HO_wavefunction(x,j)

    def momentum_operator_term(self,i,j):
        """
        Finds the term in each matrix element associated with the momentum operator
        
        Arguments:
        i(int) - the row of the term to evaluate
        j(int) - the column of the term to evaluate
        """
        if i == (j+2):
            ElementM = np.sqrt(j+1)*np.sqrt(j+2)
        elif i == j:
            ElementM = -(j+1) - j
        elif j == (i+2):
            ElementM = np.sqrt(j)*np.sqrt(j-1)
        else:
            ElementM = 0
        #set hbar = 1, omega = 1
        return ElementM/4
        
    def v_term(self, x, i, j):
        """
        Creates the function within the integral for each potential term
        
        Arguments:
        x(float) - the position to evaluate the wavefunction
        i(int) - the row of the matrix to calculate
        j(int) - the column of the matrix to calculate
        
        Returns: 
        (function) - returns the un-integrated inner product of the wavefunction with the wavefunction with the potential operator acted on it.
        """
        return self.HO_wavefunction(x,i) * self.potential(x) * self.HO_wavefunction(x,j)
        
    def potential_operator_term(self, i, j):
        """
        Finds the term in each matrix element associated with the potential operator
        
        Arguments:
        i(int) - the row of the matrix to calculate
        j(int) - the column of the matrix to calculate
        
        Returns: 
        w[0] (float) - the integrated wavefunction at a point
        """
        w = integrate.quad(self.v_term, 5*self.xmin, 5*self.xmax, (i,j))
        return w[0]

    def matrix_element_finder(self,i,j): 
        """
        Calculates the i-jth element of the matrix
        All elements are nonzero except diagonal and off-diagonal elements
        
        Arguments:
        i(int) - the row of the matrix to calculate
        j(int) - the column of the matrix to calculate
        
        Returns: 
        Element (float) - calculated ij-th element of matrix
        """
        Element =  self.momentum_operator_term(i,j) + self.potential_operator_term(i,j)
        return Element 
        
    def matrix_maker(self):
        """
        Creates a matrix and stores the values of the matrix found by Solver.matrix_element_finder as the elements of the matrix.
        """
        self.hamiltonian = np.zeros((self.n_functions,self.n_functions))
        for i in range(0, self.n_functions):
            for j in range(0, self.n_functions):
                self.hamiltonian[i][j] = self.matrix_element_finder(i,j)
        
    def matrix_solver(self):
        """
        Finds a matrix's eigenvalues and (normalized) eigenvectors
        """
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.hamiltonian)
        self.eigenvectors = np.transpose(self.eigenvectors)
        
        #NORMALIZE THE EIGENVECTORS?

def nrg_plot(psi, n, m = None):
    """
    Plots the eigenvectors and eigenvalues for a certain hamiltonian over a range of n values or at a single n value.
    
    Arguments:
    psi (Solver obj) - an object representing a specific hamiltonian
    n (int) - lower bound of eigenvectors to plot
     
    m (int) [OPTIONAL] - upper bound of eigenvectors to plot
    """
    
    #PUT THE CHANGE OF BASIS HERE INSTEAD OF IN THE SOLVER
    
    if hasattr(psi, 'transform'):
        eigenvectors = np.dot(psi.transform, psi.eigenvectors)
        eigenvectors = np.transpose(eigenvectors)
        print("eigenvectors")
        print(eigenvectors)
    else:
        #Here, may want to cut out the buffer rows of the matrix (i.e the eigenvectors with just one element)
        eigenvectors = psi.eigenvectors
        #print("eigenvectors")
    
    #The index of eigenvectors messes with the arrangement of the discrete basis solver's matrices
    if m == None:
        plt.plot(psi.xPoints,eigenvectors[n-1])
    else:
        for i in range(n,m):
            plt.plot(psi.xPoints,eigenvectors[i-1])

    plt.ylabel('WaveFunction')
    plt.xlabel('Position')
    plt.show()


def run(p_function, xmin, xmax, dim, mass, n, m = None, solver = 1, x_points = None, e_values = None, e_vectors = None, hamiltonian = None, plot = None):
    """
    Creates a solver object for a potential function and plots the potential function's wavefunction.
    
    Arguments:
    p_function (function) - a potential function
    xmin (float) - left bound of positions
    xmax (float) - right bound of positions
    dim (int) - number of increments when evaluating the wavefunctions
    mass (float) - the mass of the particle caught in our potential
    n (int) - lower bound of eigenvectors to plot
    
    m (int) [OPTIONAL] - upper bound of eigenvectors to plot
    solver (int) [OPTIONAL] - defines which solver (basis) to use:
                                (1) = Discrete Solver
                                (2) = Harmonic Oscillator Solver
    x_points (bool) [OPTIONAL] - if True, prints the xPoints array
    e_values(bool) [OPTIONAL] - if True, prints the eigenvalues array
    e_vectors(bool) [OPTIONAL] - if True, prints the eigenvectors array
    hamiltonian(bool) [OPTIONAL] - if True, prints the a array
    plot (None) [OPTIONAL] - if None, plots the wavefunction within the potential
    """
    if solver == 1:
        #note, here dim is the number of steps taken
        potential = Discrete_Solver(p_function, xmin, xmax, dim, mass)
    elif solver == 2:
        omega = 1/mass**2
        #note, here dim is the number of functions
        potential = Ho_Solver(p_function, xmin, xmax, dim, mass, omega)
        potential.HO_matrix()
    else:
        print("Change the solver variable: (1) - Discrete Solver, (2) - Harmonic Oscillator Solver")
    
    potential.matrix_maker()
    potential.matrix_solver()
    
    if x_points == True:
        print(potential.xPoints)
    
    if e_values == True:
        print(potential.eigenvalues)
        
    if e_vectors == True:
        print(potential.eigenvectors)
        
    if hamiltonian == True:
        print(potential.a)

    if plot == None:
        nrg_plot(potential, n, m)
   
   
if (__name__ == "__main__"):
    #Test Case 1: The infinite square well potential
    def square_well_potential(x):
        return 0

    #Test Case 2: The harmonic oscillator potential
    def ho_potential(x):
        return (1/2)*x**2/0.511
     

    #run(ho_potential, -1, 1, 100, 0.511, 1, x_points = True, e_values = True, e_vectors = True)
    print('buffer line')
    
    #Need to define omega as 1/mass**2
    print("harmonic oscillator")
    run(ho_potential, -1, 1, 5, 0.511, 1, solver = 2, e_vectors = True)
    
    print("square well")
    run(square_well_potential, -1, 1, 100, 0.511, 1, solver = 1, e_vectors = True)
    #w = Ho_Solver(ho_potential,-1,1,5,0.511, 1)
    #w.matrix_maker()
    #print(w.hamiltonian)
    #Random







