"""
Anne Stratman
Ben Riordan
Jan. 25th, 2018
Computational Lab in Quantum Mechanics

Solves the Schrodinger equation for time-independent potentials in a discrete or harmonic oscillator basis
Calculates matrix elements for x, x^2, p, and p^2 operators
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
        for i in range (1,self.n_steps):
            for j in range(1,self.n_steps):
                self.hamiltonian[i][j] = self.matrix_element_finder(i,j)

    def matrix_solver(self):
        """
        Finds a matrix's eigenvalues and (normalized) eigenvectors
        """
        self.eigenvalues, self.column_eigenvectors = np.linalg.eigh(self.hamiltonian)
        self.row_eigenvectors = np.transpose(self.column_eigenvectors)
        #Normalization of the eigenvectors: 1/np.sqrt(self.h)
        for i in range(0, len(self.row_eigenvectors)):
            self.row_eigenvectors[i] = (1/np.sqrt(self.h)) * self.row_eigenvectors[i]
        self.eigenvectors = np.delete(self.row_eigenvectors,[0,1],0)
        self.row_eigenvectors = np.delete(self.row_eigenvectors,[0,1],0)
        self.column_eigenvectors = np.transpose(self.row_eigenvectors)

    def xExpecValMatrix(self):
        #Padded with zeros
        #xPoints has dimensions n_steps+1
        self.positionMatrix = np.zeros((self.n_steps+1,self.n_steps+1))
        for i in range(1,self.n_steps):
            for j in range(1,self.n_steps):
                if i == j:
                    self.positionMatrix[i][j] = 1
        for i in range(self.n_steps+1):
            x_index = i
            position = self.xPoints[x_index]
            #CHECK NORMALIZATION
            if self.operator == 1:
                self.positionMatrix[x_index][x_index] = self.h * position * self.positionMatrix[x_index][x_index]
            elif self.operator == 2:
                self.positionMatrix[x_index][x_index] = self.h * position**2 * self.positionMatrix[x_index][x_index]

    def calcxExpecVal(self):
        working_matrix = np.dot(self.positionMatrix,self.column_eigenvectors)
        self.xExpecVal = np.dot(self.row_eigenvectors,working_matrix)

    def momentumElementFinder1(self,i,j):
        if i == j:
            Element = 1j/self.h
        elif i + 1 == j:
            Element = -1j/self.h
        else:
            Element = 0
        return Element

    def momentumElementFinder2(self,i,j):
        if i == j:
            Element = 2/(self.h**2)
        elif i == j + 1:
            Element = -1/(self.h**2)
        elif j == i + 1:
            Element = -1/(self.h**2)
        else:
            Element = 0
        return Element

    def pExpecValMatrix(self):
        #Padded with zeros
        self.momentumMatrix = np.zeros((self.n_steps+1,self.n_steps+1),dtype=np.complex)
        if self.operator == 1:
            for i in range(1,self.n_steps):
                for j in range(1,self.n_steps):
                    self.momentumMatrix[i][j] = self.h**2 * self.momentumElementFinder1(i,j)
        elif self.operator == 2:
            for i in range(1,self.n_steps):
                for j in range(1,self.n_steps):
                    if i == j:
                        self.momentumMatrix[i][j] = self.h**3 * self.momentumElementFinder2(i,j)
            #for i in range(len(self.xPoints)):
                #x_index = i
                #position = self.xPoints[x_index]
                #Check this - find KE terms by subtracting potential from diagonal elements of Hamiltonian
                #then find momentum terms by multiplying by 2m (KE = p^2/2m)
                #self.momentumMatrix[x_index][x_index] = 2*self.mass*(self.hamiltonian[x_index][x_index] - self.potential(position))

    def calcpExpecVal(self):
        working_matrix = np.matmul(self.momentumMatrix,self.column_eigenvectors)
        self.pExpecVal = np.matmul(self.row_eigenvectors,working_matrix)

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
    potential.matrix_maker()
    potential.matrix_solver()
    potential.xExpecValMatrix()
    potential.calcxExpecVal()
    return potential.xExpecVal

def findpExpectationValue(p_function, xmin, xmax, dim, mass, n, m, operator):
    potential = Discrete_Solver(p_function, xmin, xmax, dim, mass, operator)
    potential.matrix_maker()
    potential.matrix_solver()
    potential.pExpecValMatrix()
    potential.calcpExpecVal()
    return potential.pExpecVal

def xPlotter(potential,operator,rms=False):
    electron_mass = 511
    xValues = []
    nValues = []
    for n in range(100,1001,10):
        nValues.append(n)
        xVal = findxExpectationValue(potential,-1,1,n,electron_mass,0,10,operator)
        if rms == False:
            xValues.append(xVal[0][0])
        elif rms == True:
            xValues.append(np.sqrt(xVal[0][0]))
    plt.plot(nValues,xValues)
    plt.xlabel("Number of Dimensions")
    if rms == False:
        plt.ylabel("Expectation Value")
    elif rms == True:
        plt.ylabel("RMS Value")
    if operator == 1:
        plt.title("Expectation Value of Position for Ground State",y=1.05)
    elif operator == 2:
        if rms == False:
            plt.title("Expectation Value of Position Squared for Ground State",y=1.05)
        else:
            plt.title("RMS Position for Ground State",y=1.05)
    plt.show()

def pPlotter(potential,operator,rms=False):
    electron_mass = 511
    pValues = []
    nValues = []
    for n in range(100,1001,20):
        nValues.append(n)
        pVal = findpExpectationValue(potential,-1,1,n,electron_mass,0,10,operator)
        if operator == 1:
            momentum = np.imag(pVal[0][0])
        elif operator == 2:
            momentum = np.real(pVal[0][0])
        if rms == False:
            pValues.append(momentum)
        elif rms == True:
            pValues.append(np.sqrt(momentum))
    plt.plot(nValues,pValues)
    plt.xlabel("Number of Dimensions")
    if rms == False:
        plt.ylabel("Expectation Value")
    elif rms == True:
        plt.ylabel("RMS Value")
    if operator == 1:
        plt.title("Expectation Value of Momentum for Ground State",y=1.05)
    elif operator == 2:
        if rms == False:
            plt.title("Expectation Value of Momentum Squared for Ground State",y=1.05)
        else:
            plt.title("RMS Momentum for Ground State",y=1.05)
    plt.show()

if __name__ == "__main__":

    #Numbers equal to zero in the current precision print as zero
    np.set_printoptions(suppress=True)

    electron_mass = 511
    omega = 1

    def square_well_potential(x):
        return 0
    square_well = (square_well_potential, "Square Well")

    def ho_potential(x):
        return (1/2)*electron_mass*(omega**2)*(x**2) 
    ho = (ho_potential,"Harmonic Oscillator")

    def trig_potential(x):
        return (np.sin(x))**2
    trig = (trig_potential,"Sine Squared")

    #run(square_well, -0.3, 0.3, 100, electron_mass, 0, 5, solver = 1, energy = True, hamiltonian = True)
    #plt.show()
    #print(findxExpectationValue(square_well,-0.3,0.3,500,electron_mass,0,5,1))
    #print(findxExpectationValue(square_well,-0.3,0.3,500,electron_mass,0,5,2))
    #print(findpExpectationValue(square_well,-0.3,0.3,5,electron_mass,0,5,1))
    #print(findpExpectationValue(square_well,-0.3,0.3,5,electron_mass,0,5,2))

    #run(square_well,-0.1,0.1,100,electron_mass,0,5)
    #plt.show()

    #xPlotter(square_well,1)
    #xPlotter(square_well,2)
    #xPlotter(square_well,2,rms=True)
    #pPlotter(square_well,1)
    #pPlotter(square_well,2)
    #pPlotter(square_well,2,rms=True)

    #xPlotter(ho,1) 
    #xPlotter(ho,2)  
    #xPlotter(ho,2,rms=True)
    #pPlotter(ho,1)
    #pPlotter(ho,2)
    #pPlotter(ho,2,rms=True)

    #xPlotter(trig,1)
    #xPlotter(trig,2)
    #xPlotter(trig,2,rms=True)
    #pPlotter(trig,1)
    #pPlotter(trig,2)
    #pPlotter(trig,2,rms=True)









