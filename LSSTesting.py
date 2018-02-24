import numpy as np
import scipy.misc
import scipy.integrate as integrate
import hermite

electron_mass = 0.511

class Ho_Solver:
    def __init__(self, potential, xmin, xmax, n_functions, particle_mass, omega):
        self.potential = potential
        self.xmin = xmin
        self.xmax = xmax
        self.n_steps = 100 #want to be rows
        self.n_functions = n_functions #want to be columns
        self.mass = particle_mass
        self.omega = omega
        self.h_bar = 1

        self.h = (self.xmax-self.xmin)/self.n_steps
        
        self.xPoints = np.zeros(self.n_steps)
        for i in range(self.n_steps):
            self.xPoints[i] = i*self.h + self.xmin

    def HO_wavefunction(self,x,n):
        """
        Defines the harmonic oscillator wavefunction
        """
        curvy_e = np.sqrt(self.mass*self.omega/self.h_bar)*x

        Hermite = hermite.hermite(n,curvy_e)
        psi = (self.mass*self.omega/(np.pi*self.h_bar))**(1/4) * 1/(np.sqrt((2**n)*scipy.misc.factorial(n))) * Hermite * np.exp(-curvy_e**2/2)

    def integrand(self,x,i,j):
        """
        Defines the potential integrand
        """
        curvy_e = np.sqrt(self.mass*self.omega/self.h_bar)*x

        Hermite1 = hermite.hermite(i,curvy_e)
        psi1 = (self.mass*self.omega/(np.pi*self.h_bar))**(1/4) * 1/(np.sqrt((2**i)*scipy.misc.factorial(i))) * Hermite1 * np.exp((-curvy_e**2)/2)

        Hermite2 = hermite.hermite(j,curvy_e)
        psi2 = (self.mass*self.omega/(np.pi*self.h_bar))**(1/4) * 1/(np.sqrt((2**j)*scipy.misc.factorial(j))) * Hermite2 * np.exp((-curvy_e**2)/2)

        potential = (1/2) * self.mass * self.omega * (x**2)

        return psi1 * potential * psi2

    def potential_integral(self,i,j):
        print("i = ", i, "j = ", j)
        potential_value = integrate.quad(self.integrand,self.xmin,self.xmax,args=(i,j))
        print("potential=",potential_value[0])
        return potential_value[0]

    def momentum_operator_term(self,i,j):
        """
        Finds the term in each matrix element associated with the momentum operator
        
        Arguments: i is row and j is column
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
        #set hbar = 1, omega = 1
        return prefactor*ElementM

    def matrix_element_finder(self,i,j): 
        """
        Calculates the i-jth element of the matrix
        All elements should be nonzero except diagonal and off-diagonal elements
        """
        Element =  self.momentum_operator_term(i,j) + self.potential_integral(i,j)
        return Element

    def matrix_maker(self):
        """
        Creates a matrix and stores the values of the matrix found by Solver.matrix_element_finder as the elements of the matrix.
        """
        self.hamiltonian = np.zeros((self.n_functions,self.n_functions))
        for i in range(0, self.n_functions):
            for j in range(0, self.n_functions):
                self.hamiltonian[i][j] = self.matrix_element_finder(i,j)
        return self.hamiltonian

    def matrix_solver(self):
        """
        Finds a matrix's eigenvalues and (normalized) eigenvectors
        """
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.hamiltonian)
        self.eigenvectors = np.transpose(self.eigenvectors)

    #Start non-square matrix stuff here

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

def run(p_function, xmin, xmax, n_functions, mass, omega):

    test = Ho_Solver(p_function, xmin, xmax, n_functions, mass, omega)
    i = n_functions
    j = n_functions
    print(test.matrix_maker())




if __name__ == "__main__":
    electron_mass = 511
    omega = 1

    def ho_potential(x):
        return (1/2)*electron_mass*(omega**2)*(x**2)

    run(ho_potential,-1,1,5,511,1)


