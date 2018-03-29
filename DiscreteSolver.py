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
    def __init__(self, potential, xmin, xmax, n_steps, particle_mass, operator = None):
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

        self.operator = operator

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
        
    

    def xExpecValMatrix(self):
        #Need to pad with zeros
        self.positionMatrix = np.zeros((self.n_steps+1,self.n_steps+1))
        for i in range(1,self.n_steps):
            for j in range(1,self.n_steps):
                if i == j:
                    self.positionMatrix[i][j] = 1
        for i in range(len(self.xPoints)):
            x_index = i
            position = self.xPoints[x_index]
            if self.operator == 1:
                self.positionMatrix[x_index][x_index] = position * self.positionMatrix[x_index][x_index]
            elif self.operator == 2:
                self.positionMatrix[x_index][x_index] = position**2 * self.positionMatrix[x_index][x_index]

    def momentumElementFinder(self,i,j):
        if i == j:
            Element = 1j/self.h
        elif i + 1 == j:
            Element = -1j/self.h
        else:
            Element = 0
        return Element

    def pExpecValMatrix(self):
        #Need to pad with zeros?
        self.momentumMatrix = np.zeros((self.n_steps+1,self.n_steps+1),dtype=np.complex)
        if self.operator == 1:
            for i in range(1,self.n_steps):
                for j in range(1,self.n_steps):
                    self.momentumMatrix[i][j] = self.momentumElementFinder(i,j)
        elif self.operator == 2:
            for i in range(self.n_steps+1):
                for j in range(self.n_steps+1):
                    if i == j:
                        self.momentumMatrix[i][j] = 1
            for i in range(len(self.xPoints)):
                x_index = i
                position = self.xPoints[x_index]
                #Check this - find KE terms by subtracting potential from diagonal elements of Hamiltonian
                #then find momentum terms by multiplying by 2m (KE = p^2/2m)
                self.momentumMatrix[x_index][x_index] = 2*self.mass*(self.hamiltonian[x_index][x_index] - self.potential(position))
    

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

def run(p_function, xmin, xmax, dim, mass, n, m, operator = None, energy = None, solver = 1, x_points = None, e_values = None, e_vectors = None, hamiltonian = None, plot = None):
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

def findxExpectationValue(p_function, xmin, xmax, dim, mass, n, m, operator):
    potential = Discrete_Solver(p_function, xmin, xmax, dim, mass, operator)
    potential.xExpecValMatrix()
    return potential.positionMatrix

def findpExpectationValue(p_function, xmin, xmax, dim, mass, n, m, operator):
    potential = Discrete_Solver(p_function, xmin, xmax, dim, mass, operator)
    potential.matrix_maker()
    potential.pExpecValMatrix()
    return potential.momentumMatrix

if __name__ == "__main__":

    electron_mass = 511

    def square_well_potential(x):
        return 0
    square_well = (square_well_potential, "Square Well")

    print(findxExpectationValue(square_well,-0.1,0.1,100,electron_mass,0,5,1))
    print(findxExpectationValue(square_well,-0.1,0.1,100,electron_mass,0,5,2))
    print(findpExpectationValue(square_well,-0.1,0.1,100,electron_mass,0,5,1))
    print(findpExpectationValue(square_well,-0.1,0.1,100,electron_mass,0,5,2))

    #run(square_well,-0.1,0.1,100,electron_mass,0,5)
    #plt.show()









