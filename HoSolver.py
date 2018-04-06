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

    def set_omega(self,omega):
        self.omega = omega
        
    #IMPORTANT WAVEFUNCTION AND MATRIX
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

    #CALCULATE HAMILTONIAN TERMS
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
        
        return psi1 * self.potential(x) * psi2
    
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
    
    #ASSEMBLE THE HAMILTONIAN MATRIX
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
    
    #EXPECTATION VALUES
    def expectation_position(self):
        """
        Calculates the expetation value of the position for an eigenvector in the solver basis.
        """
        v = self.potential
        def pot(x): return x
        self.potential = pot

        self.pos_exp = np.zeros((self.n_functions,self.n_functions))
        for i in range(0,self.n_functions):
            for j in range(0, self.n_functions):
                self.pos_exp[i][j] = self.potential_operator_term(i,j)
                
        self.potential = v
        
    def expectation_position2(self):
        """
        Calculates the expectation value of the square of the position for an eigenvector in the solver basis.
        """
        v = self.potential
        def pot(x): return x**2
        self.potential = pot

        self.pos2_exp = np.zeros((self.n_functions,self.n_functions))
        for i in range(0,self.n_functions):
            for j in range(0, self.n_functions):
                self.pos2_exp[i][j] = self.potential_operator_term(i,j)
                
        self.potential = v
    
    def expectation_momentum(self):
        """
        Calculates the expectastion value of the momentum for an eigenvector in the solver basis.
        p = i*sqrt(hbar*m*omega/2)*(a_+ - a_-)
        """
        const = np.sqrt(self.h_bar * self.mass * self.omega / 2)
        self.mom_exp = np.zeros((self.n_functions,self.n_functions))
        for i in range(0, self.n_functions):
            for j in range(0, self.n_functions):
                if i == j+1:
                    self.mom_exp[i][j] = np.sqrt(i)*const
                elif i == j-1:
                    self.mom_exp[i][j] = np.sqrt(j)*const
                else:
                    self.mom_exp[i][j] = 0
    
    def expectation_momentum2(self):
        """
        Calculates the expectation value of the momentum squared for an eigenvecor in the solver basis.
        """
        #WRITE MOMENTUM IN TERMS OF THIS (MAKE THIS THE CONDITIONAL STATEMENTS IN MOMENTUM HERE AND CALL THIS IN MOMENTUM)
        self.mom2_exp = np.zeros((self.n_functions, self.n_functions))
        
        for i in range(0,self.n_functions):
            for j in range(0, self.n_functions):
                self.mom2_exp[i][j]  = np.abs(self.momentum_operator_term(i,j))
                
        self.mom2_exp = self.mom2_exp * (2*self.mass)
        
        
def nrg_plot(psi, solver, n, m = None, energy = False):
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

def exp_plot(solver_obj,exp_type, tran_type, plot_type):
    """Plots the expectation value of a particle within a potential against the 
    """
    if exp_type == 1:
        name = "x"
        
    elif exp_type == 2:
        name = "x^2"
        
    elif exp_type == 3:
        name = "p"
        
    elif exp_type == 4:
        name = "p^2"
        
    else:
        print("Please set \'operator\' equal to:")
        print("1 - position expectation value")
        print("2 - position squared expectation value")
        print("3 - momentum expectation value")
        print("4 - momentum squared expectation value")
    
    expectation_values = []
    #Plotting vs number of basis functions
    if plot_type == 1:
        #plotting transition amplitudes
        if tran_type == True:
            title = "Transition probability of " + name + " vs. number of functions"
            y_label = "Transition probability of " + name
            
            #need to transition by 2 to be non-zero
            if exp_type == 2 or exp_type == 4:
                n_range = range(2,solver_obj.n_functions-1)
                for n in n_range:
                    expectation_values.append(expectation((solver_obj.potential,solver_obj.potential_name),solver_obj.xmin,solver_obj.xmax,n,solver_obj.mass,0,solver_obj.omega,operator = exp_type, transition = tran_type))
                    #print("Number of functions: " + str(n))
                    print(expectation_values[n-2])
                    #print()
               
            #need to transition by 1 to be non-zero
            elif exp_type == 1 or exp_type == 3:
                n_range = range(1,solver_obj.n_functions-1)
                for n in n_range:
                    expectation_values.append(expectation((solver_obj.potential,solver_obj.potential_name),solver_obj.xmin,solver_obj.xmax,n,solver_obj.mass,0,solver_obj.omega,operator = exp_type, transition = tran_type))
                    #print("Number of functions: " + str(n))
                    print(expectation_values[n-1])
                    #print()
        
        #plotting expectation values
        else:
            n_range = range(1,solver_obj.n_functions)
            title = "Expectation value of " + name + " vs. number of functions"
            y_label = "Expectation value of " + name
            for n in n_range:
                expectation_values.append(expectation((solver_obj.potential,solver_obj.potential_name),solver_obj.xmin,solver_obj.xmax,n,solver_obj.mass,0,solver_obj.omega,operator = exp_type, transition = tran_type))
                #print("Number of functions: " + str(n))
                print(expectation_values[n-1])
                #print()
            
        plt.plot(n_range, expectation_values, label = 'hbar*omega = ' + str(solver_obj.omega))
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel("Number of Functions")
    
    #Plotting vs hbar*omega
    elif plot_type == 2:
        #plotting transition amplitudes
        if tran_type == True:
            title = "Transition probability of " + name + " vs. hbar*omega"
            y_label = "Transition probability of " + name
            
        #plotting expectation values
        else:
            title = "Expectation value of " + name + " vs. hbar*omega"
            y_label = "Expectation value of " + name
            
        #arbitrary list of omegas
        omega_list = [0.25,0.5,1,5,10]
        for n in range(len(omega_list)):
            expectation_values.append(expectation((solver_obj.potential,solver_obj.potential_name),solver_obj.xmin,solver_obj.xmax,solver_obj.n_functions,solver_obj.mass,0,omega_list[n],operator = exp_type, transition = tran_type))
            #print("hbar*omega: " + str(omega_list[n]))
            print(expectation_values[n])
            
        plt.plot(omega_list, expectation_values)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel("hbar*omega")
        
    #Plotting sqrt(Expectation values) vs. number of basis functions
    elif plot_type == 3:
        #plotting transition amplitudes
        if tran_type == True:
            n_range = range(2,solver_obj.n_functions-1)
            title = "Transition probability of " + name + " vs. number of functions"
            y_label = "Transition probability of " + name
            for n in n_range:
                expectation_values.append(np.sqrt(np.abs(expectation((solver_obj.potential,solver_obj.potential_name),solver_obj.xmin,solver_obj.xmax,n,solver_obj.mass,0,solver_obj.omega,operator = exp_type, transition = tran_type))))
                #print("Number of functions: " + str(n))
                print(expectation_values[n-2])
                #print()
                
        #plotting expectation values
        else:
            n_range = range(1,solver_obj.n_functions)
            title = "Expectation value of " + name + " vs. number of functions"
            y_label = "Expectation value of " + name
            for n in n_range:
                expectation_values.append(np.sqrt(np.abs(expectation((solver_obj.potential,solver_obj.potential_name),solver_obj.xmin,solver_obj.xmax,n,solver_obj.mass,0,solver_obj.omega,operator = exp_type, transition = tran_type))))
                #print("Number of functions: " + str(n))
                print(expectation_values[n-1])
                #print()
            
        plt.plot(n_range, expectation_values, label = 'hbar*omega = ' + str(solver_obj.omega))
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel("Number of Functions")

    plt.axis("tight")
            
def multi_plot(solver_obj, exp_type, tran_type, plot_type, multi_type = 1):
    if multi_type == 1:
        omega_list = [0.25,0.5,1,5,10]
        for n in range(len(omega_list)):
            solver_obj.set_omega(omega_list[n])
            exp_plot(solver_obj,exp_type,tran_type,plot_type)
        #FIGURE OUT HOW TO ADD A LEGEND BELOW
        ax = plt.gca()
        ax.legend(loc = 'best')

def run(p_function, xmin, xmax, dim, mass, n, m = None, energy = None, solver = 2, x_points = None, e_values = None, e_vectors = None, hamiltonian = None, plot_psi = None, plot_exp = (None, 1, None, 1), plot_multi = None):
    """
    Creates a solver object for a potential function and plots the potential function's wavefunction.
    
    Arguments:
    p_function (tuple) - (a potential function
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

    if plot_psi == True:
        nrg_plot(potential, solver, n, m, energy)
        
    if plot_multi == None and plot_exp[0] == True:
        exp_plot(potential,plot_exp[1], plot_exp[2], plot_exp[3])
        
    if plot_multi != None and plot_exp[0] == True:
        multi_plot(potential,plot_exp[1], plot_exp[2], plot_exp[3], multi_type = plot_multi)
        
        
def expectation(p_function, xmin, xmax, dim, mass, n, omega, solver = 2, operator = 0, transition = None):
    """Calculates expectation values for different physical values for a particle within a certain potential function
    
    Arguments:
        operator (int) - defines which expectation value to calculate:
                        (1) - position
                        (2) - position^2
                        (3) - momentum
                        (4) - momentum^2
        transition (int) - calculates the 'transition amplitudes'
    """
    if solver == 2:
        #note, here dim is the number of functions
        potential = Ho_Solver(p_function, xmin, xmax, dim, mass, omega)
        potential.HO_matrix()
    
    potential.matrix_maker()
    potential.matrix_solver()
    
    e_vectors = potential.eigenvectors
    e_values = potential.eigenvalues
    
    if transition == True and (operator == 1 or operator == 3):
        m = n-1
    elif transition == True and (operator == 2 or operator == 4):
        m = n-2
    else: 
        m = n
    
    if operator == 1:
        potential.expectation_position()
        op_matrix = potential.pos_exp
        #print(op_matrix)
        expectation = np.dot(np.transpose(e_vectors[n]),np.dot(op_matrix,e_vectors[m]))
        
    elif operator == 2:
        potential.expectation_position2()
        op_matrix = potential.pos2_exp
        #print(op_matrix)
        expectation = np.dot(np.transpose(e_vectors[n]),np.dot(op_matrix,e_vectors[m]))
    
    elif operator == 3:
        potential.expectation_momentum()
        op_matrix = potential.mom_exp
        #print(op_matrix)
        expectation = np.dot(np.transpose(e_vectors[n]),np.dot(op_matrix,e_vectors[m]))
    
    elif operator == 4:
        potential.expectation_momentum2()
        op_matrix = potential.mom2_exp
        #print(op_matrix)
        expectation = np.dot(np.transpose(e_vectors[n]),np.dot(op_matrix,e_vectors[m]))
    
    else:
        return
    
    return expectation
    

if __name__ == "__main__":
    electron_mass = 511
    omega = 1

    def ho_potential(x):
        return (1/2)*electron_mass*(omega**2)*(x**2) 
    ho = (ho_potential,"Harmonic Oscillator")
    
    def ho_bump_potential(x):
        return x**4 - x**2
    bump = (ho_bump_potential,"Perturbed Harmonic Oscillator")
    
    def plotty_mcplotface(potential, plot_type):
        """
        (1) - x expectation
        (2) - x^2 expectation
        (3) - p expectation
        (4) - p^2 expectation
        (5) - x transition
        (6) - x^2 transition
        (7) - p transition
        (8) - p^2 transition
        (9) - square root of x^2 expectation
        (10) - square root of p^2 expectation
        """
        #EXPECTATIONS
        if plot_type == 1:
            #Plots the expectation value of x against the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 10, electron_mass, 0, solver = 2, plot_exp = (True,1, False, 1),plot_multi = True)
    
        if plot_type == 2:
            #Plots the expectation value of x^2 against the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 10, electron_mass, 0, solver = 2, plot_exp = (True,2, False, 1),plot_multi = True)
    
        if plot_type == 3:
            #Plots the expectation value of p against the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 10, electron_mass, 0, solver = 2, plot_exp = (True,3, False, 1),plot_multi = True)
    
        if plot_type == 4:
            #Plots the expectation value of p^2 against the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 10, electron_mass, 0, solver = 2, plot_exp = (True,4, False, 1),plot_multi = True)
    
    
        #TRANSITIONS
        if plot_type == 5:
            #Plots the transition probability of x agains the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 11, electron_mass, 0, solver = 2, plot_exp = (True,1, True, 1),plot_multi = True)
    
        if plot_type == 6:
            #Plots the transition probability of x^2 agains the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 12, electron_mass, 0, solver = 2, plot_exp = (True,2, True, 1),plot_multi = True)
    
        if plot_type == 7:
            #Plots the transition probability of p agains the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 11, electron_mass, 0, solver = 2, plot_exp = (True,3, True, 1),plot_multi = True)
    
        if plot_type == 8:
            #Plots the transition probability of p^2 agains the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 12, electron_mass, 0, solver = 2, plot_exp = (True,4, True, 1),plot_multi = True)
            
            
        #ROOTS OF EXPECTATIONS
        if plot_type == 9:
            #Plots the expectation value of x^2 against the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 10, electron_mass, 0, solver = 2, plot_exp = (True,2, False, 3),plot_multi = True)
            
        if plot_type == 10:
            #Plots the expectation value of p^2 against the number of basis functions for multiple values of hbar*omega
            run(potential,-0.3, 0.3, 10, electron_mass, 0, solver = 2, plot_exp = (True,4, False, 3),plot_multi = True)
    
    #Plots for the Harmonic Oscillator Potential:
    #These two converge, but do encounter negative values? ->
    #plotty_mcplotface(ho,5)
    #plotty_mcplotface(ho,6)
    
    #These two diverge->
    #plotty_mcplotface(ho,9)
    plotty_mcplotface(ho,9)
    
    #Plots for the Perturbed Harmonic Oscillator Potential:
    #plotty_mcplotface(bump,5)
    #plotty_mcplotface(bump,6)
    #plotty_mcplotface(bump,9)
    #plotty_mcplotface(bump,10)
    
    plt.legend(loc = 'best')
    plt.show()
    
    


