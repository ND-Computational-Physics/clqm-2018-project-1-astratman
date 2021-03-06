"""
Anne Stratman
Ben Riordan
Jan. 25th, 2018
Computational Lab in Quantum Mechanics

Solves the Schrodinger equation for time-independent potentials in a discrete or harmonic oscillator basis
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.misc
import scipy.special
import hermite
          
class Discrete_Solver:
    def __init__(self, potential, xmin, xmax, n_steps, particle_mass):
        """
        Arguments:
        \Initialized\
        self (obj)
        potential(tuple (function, string)) - potential function to use in solving the LSE, along with its name
        xmin(float) - left bound of position
        xmax(float) - right bound of position
        n_steps(int) - -number of increments in interval
        particle_mass (float) - mass of particle in keV
        
        \Assigned\
        h(float) - the spacing between each x point
        xPoints(float array) - a 1D array of x points
        hamiltonian(float array) - The hamiltonian operator matrix (2D array)
        eigenvalues(float array) - a 1D array of the eigenvalues for our potential
        eigenvectors(float array) - a 2D array of eigenevectors for our potential
        """
        self.potential = potential[0]
        self.potential_name = potential[1]
        
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
        self.hamiltonian = np.zeros((self.n_steps+1,self.n_steps+1))
        for i in range(1, self.n_steps):
            for j in range(1, self.n_steps):
                self.hamiltonian[i][j] = self.matrix_element_finder(i,j)

    def matrix_solver(self):
        """
        Finds a matrix's eigenvalues and (normalized) eigenvectors
        """
        self.eigenvalues, work_eigenvectors = np.linalg.eigh(self.hamiltonian)
        work_eigenvectors = np.transpose(work_eigenvectors)
        
        #Normalization of the eigenvectors: 1/np.sqrt(self.h)
        for i in range(0, len(work_eigenvectors)):
            work_eigenvectors[i] = (1/np.sqrt(self.h)) * work_eigenvectors[i]
        self.eigenvectors = np.delete(work_eigenvectors,[0,1],0)

            
class Ho_Solver:
    def __init__(self, potential, xmin, xmax, n_functions, particle_mass, omega):
        """
        Attributes:
        \Initialized\
        self (obj)
        potential(tuple (function, string)) - potential function to use in solving the LSE, along with its name
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
        self.potential = potential[0]
        self.potential_name = potential[1]
        
        self.xmin = xmin
        self.xmax = xmax
        self.n_steps = 100 #want to be rows
        self.n_functions = n_functions #want to be columns
        
        self.mass = particle_mass
        self.omega = omega
        self.pi = np.pi
        self.h_bar = 1

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
        curvy_e = np.sqrt(self.mass*self.omega/self.h_bar)*x

        Hermite = hermite.hermite(n,curvy_e)
        psi = (self.mass*self.omega/(np.pi*self.h_bar))**(1/4) * 1/(np.sqrt((2**n)*scipy.misc.factorial(n))) * Hermite * np.exp(-(curvy_e**2)/2)
        return psi

    def HO_matrix(self):
        """
        Creates a matrix with different harmonic oscillator wavefunctions evaluated at a vector of xPoints to use for a change of basis
        
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
        prefactor = (-1*self.h_bar*self.omega)/4
        if i == (j+2):
            ElementM = np.sqrt(j+1)*np.sqrt(j+2)
        elif i == j:
            ElementM = -(j+1) - j
        elif j == (i+2):
            ElementM = np.sqrt(j)*np.sqrt(j-1)
        else:
            ElementM = 0
        return prefactor*ElementM

    def integrand(self,x,i,j):
        """
        Creates the function within the integral for each potential term
        
        Arguments:
        x(float) - the position to evaluate the wavefunction
        i(int) - the row of the matrix to calculate
        j(int) - the column of the matrix to calculate
        
        Returns: 
        (function) - returns the un-integrated inner product of the wavefunction with the wavefunction with the potential operator acted on it.
        """
        curvy_e = np.sqrt(self.mass*self.omega/self.h_bar)*x

        Hermite1 = hermite.hermite(i,curvy_e)
        psi1 = (self.mass*self.omega/(np.pi*self.h_bar))**(1/4) * 1/(np.sqrt((2**i)*scipy.misc.factorial(i))) * Hermite1 * np.exp(-(curvy_e**2)/2)

        Hermite2 = hermite.hermite(j,curvy_e)
        psi2 = (self.mass*self.omega/(np.pi*self.h_bar))**(1/4) * 1/(np.sqrt((2**j)*scipy.misc.factorial(j))) * Hermite2 * np.exp(-(curvy_e**2)/2)

        potential = self.potential(x)

        return psi1 * potential * psi2
    
    def potential_operator_term(self, i, j):
        """
        Finds the term in each matrix element associated with the potential operator
        
        Arguments:
        i(int) - the row of the matrix to calculate
        j(int) - the column of the matrix to calculate
        
        Returns: 
        w[0] (float) - the integrated wavefunction at a point
        """
        potential_value = integrate.quad(self.integrand,-np.inf,np.inf,args=(i,j))
        return potential_value[0]

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
        

def nrg_plot(psi, solver, n, m, energy = False):
    """
    Plots the eigenvectors and eigenvalues for a certain hamiltonian over a range of n values or at a single n value.
    
    Arguments:
    psi (Solver obj) - an object representing a specific hamiltonian
    n (int) - lower bound of eigenvectors to plot
     
    m (int) - upper bound of eigenvectors to plot
    energy(bool) [OPTIONAL] - plot the energy eigenvalues vs n (quantum number) values if True
    """
    #Transform (if necessary) and plot wavefunctions
    if hasattr(psi, 'transform'):
        eigenvectors = np.dot(psi.transform, psi.eigenvectors)
        eigenvectors = np.transpose(eigenvectors)
    else:
        eigenvectors = psi.eigenvectors
    
    if energy == True:
        if solver == 1:
            nPoints = []
            e_values = []
            for i in range(n,m+1):
                nPoints.append(i)
                e_values.append(psi.eigenvalues[i])
            plt.plot(nPoints, e_values)
            
            #nPoints = []
            #for i in range(len(psi.xPoints)):
            #    nPoints.append(i)
            #plt.plot(nPoints, psi.eigenvalues)
            
            plt.title(psi.potential_name + " Eigenvalues - Discrete Basis")
            plt.xlabel('n')
            plt.ylabel('Energy')
            #plt.show()
        elif solver == 2:
            nPoints = []
            e_values = []
            for i in range(n,m+1):
                nPoints.append(i)
                e_values.append(psi.eigenvalues[i])
            plt.plot(nPoints, e_values)
            
            #nPoints = []
            #for i in range(psi.n_functions):
            #    nPoints.append(i)
            #plt.plot(nPoints, psi.eigenvalues)
            
            plt.title(psi.potential_name + " Eigenvalues - Harmonic Oscillator Basis")
            plt.xlabel('n')
            plt.ylabel('Energy')
            #plt.show()   
    else:
        if m == None:
            plt.plot(psi.xPoints,eigenvectors[n])
            name = "n = " + str(n) + " Solution to the NLSE for the " + psi.potential_name + " Potential"

        else:
            for i in range(n,m+1):
                plt.plot(psi.xPoints,eigenvectors[i])
            name = "n = " + str(n) + " - " + str(m) + " Solutions to the NLSE for the " + psi.potential_name + " Potential"   
    
        plt.title(name)
        plt.ylabel('Wavefunction')
        plt.xlabel('Position')
        plt.axis('tight')
        #plt.show()


def run(p_function, xmin, xmax, dim, mass, n, m, energy = None, solver = 1, x_points = None, e_values = None, e_vectors = None, hamiltonian = None, plot = None):
    """
    Creates a solver object for a potential function and plots the potential function's wavefunction.
    
    Arguments:
    p_function (function) - a potential function
    xmin (float) - left bound of positions
    xmax (float) - right bound of positions
    dim (int) - number of increments when evaluating the wavefunctions
    mass (float) - the mass of the particle caught in our potential
    n (int) - lower bound of eigenvectors to plot
    
    m (int) - upper bound of eigenvectors to plot
    solver (int) [OPTIONAL] - defines which solver (basis) to use:
                                (1) = Discrete Solver
                                (2) = Harmonic Oscillator Solver
    x_points (bool) [OPTIONAL] - if True, prints the xPoints array
    e_values(bool) [OPTIONAL] - if True, prints the eigenvalues array
    e_vectors(bool) [OPTIONAL] - if True, prints the eigenvectors array
    hamiltonian(bool) [OPTIONAL] - if True, prints the a array
    plot (None) [OPTIONAL] - if None, plots the wavefunction within the potential
    """
    if n > dim-1 or ( m != None and m > dim-1 ):
        print("The value of \'n\' must be less than the value of \'dim-1\'.")
        return
    
    if solver == 1:
        #note, here dim is the number of steps taken
        potential = Discrete_Solver(p_function, xmin, xmax, dim, mass)
    elif solver == 2:
        omega = 1
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
        print(potential.hamiltonian)

    if plot == None:
        nrg_plot(potential, solver, n, m, energy)

   
if (__name__ == "__main__"):
    electron_mass = 511
    omega = 1

    #Test Case 1: The infinite square well potential
    def square_well_potential(x):
        return 0
    square_well = (square_well_potential, "Square Well")

    
    #Test Case 2: The harmonic oscillator potential
    def ho_potential(x):
        return (1/2)*electron_mass*(omega**2)*(x**2) 
    ho = (ho_potential,"Harmonic Oscillator")

    #Test Case 3: The tangent potential
    def tangent_potential(x):
        return np.tan(x)
    tan = (tangent_potential,"Tangent")
    
    
    #Run Test Cases
    
    #NOTE: for these first sets of test cases, uncomment the commented lines in nrg_plot and comment the lines above them within the conditional statement.

    #Eigenvalue Plots
    #for i in range(10,200,20):
    #    run(square_well, -0.3, 0.3, i, electron_mass, 0, solver = 1, energy = True)
    #plt.show()
    #for i in range(5,10):
    #    run(square_well, -0.3, 0.3, i, electron_mass, 0, solver = 2, energy = True)
    #plt.show()
    #for i in range(10,200,20):
    #    run(ho, -0.3, 0.3, i, electron_mass, 0, solver = 1, energy = True)
    #plt.show()
    #for i in range(5,10):
    #    run(ho, -0.3, 0.3, i, electron_mass, 0, solver = 2, energy = True)
    #plt.show()
    #run(tan, -0.3, 0.3, 100, electron_mass, 1, m=5, solver = 1, plot = True)
    #run(tan, -0.3, 0.3, 10, electron_mass, 0, m=5, solver = 2, plot = True)

    #Point Density Plot
    #for i in range(1,10):
    #    run(ho,-0.01*i, 0.01*i, 100, electron_mass, 0, solver = 1)
    #plt.show()
    
    
    #NOTE: for these test cases, they will run with the current code
    
    #Plot energy v. n of Square well potential in Discrete Solver
    #run(ho, -0.3, 0.3, 200, electron_mass, 0, m=100, solver = 1, energy = True)
    
    #Plot energy v. n of HO potential in Discrete Solver
    #run(ho, -0.3, 0.3, 200, electron_mass, 0, m=100, solver = 1, energy = True)
    
    #Plot energy v. n of HO potential in HO Solver
    #run(ho, -0.3, 0.3, 20, electron_mass, 0, m=19, solver = 2, energy = True)
    
    #Plot n=0 HO wavefunction from HO Solver
    #run(ho,-0.3, 0.3, 10, electron_mass, 0, solver = 2)
    
    #Plot n=0 HO wavefunction from Discrete Solver
    #run(ho,-0.3, 0.3, 100, electron_mass, 0, solver = 1)
    
    #Plot n=30 HO wavefunction from Discrete Solver
    #run(ho,-0.3, 0.3, 100, electron_mass, 30, solver = 1)
    
    #Plot HO wavefunctions from HO Solver
    #run(ho, -0.3, 0.3, 10, electron_mass, 0, m=2, solver = 2)
    
    #Plot HO wavefunctions from Discrete Solver
    #run(ho, -0.3, 0.3, 100, electron_mass, 0, m=2, solver = 1)
    
    #Plot n=0 Square well wavefunction from Discrete Solver
    #run(square_well,-0.3, 0.3, 100, electron_mass, 0, solver = 1)
    
    #Plot n=0 Square well wavefunction from HO Solver
    #run(square_well,-0.3, 0.3, 100, electron_mass, 0, solver = 1)
    
    #Plot Square well wavefunctions from Discrete Solver
    #run(square_well,-0.3, 0.3, 100, electron_mass, 0, m=2, solver = 1)
    
    plt.show()


