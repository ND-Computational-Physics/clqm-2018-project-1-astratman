""" hermite.py -- evaluate hermite polynomial by recurrence

    Language: Python 3
    Mark A. Caprio
    University of Notre Dame
    Written for Computational Methods in Physics, Spring 2016.
    3/15/16 (mac) : Created using code from ho_expectation.py.
"""

import math
import numpy as np
import time

################################################################
# wave function definition
################################################################

def hermite_recursive(x,n):
    """ Evaluate Hermite polynomial H_n(x) by recurrence.

    Naive implementation through recursive function calls.

    Uses recurrence H(n,x)=2*x*H(n-1,x)-2*(n-1)*H(n-2,x).
    Obtained from Newman recurrence by shifting n -> n-1.

    Arguments:
        n (int) : order of Hermite polynomial (nonnegative)
        x (float) : argument of Hermite polynomial

    Returns:
        (float) : value of Hermite polynomial
    """

    # diagnostic counter
    ## global times_called
    ## times_called += 1

    # obtain value as seed value or by recursive evaluation
    if (n==0):
        value = 1
    elif (n==1):
        value = 2*x
    else:
        value = 2*x*hermite_recursive(x,n-1) - 2*(n-1)*hermite_recursive(x,n-2)

    return value

def hermite_recursive_gutted(x,n):
    """Evaluate Hermite polynomial H_n(x) by recurrence (timing test).

    Gutted to remove all arithmetic (or at least all multiplications).
    Time is presumably therefore from the "overhead" of the function
    calls.

    Uses recurrence H(n,x)=2*x*H(n-1,x)-2*(n-1)*H(n-2,x).
    Obtained from Newman recurrence by shifting n -> n-1.

    Arguments:
        n (int) : order of Hermite polynomial (nonnegative)
        x (float) : argument of Hermite polynomial

    Returns:
        (float) : value of Hermite polynomial
    """

    # obtain value as seed value or by recursive evaluation
    if (n==0):
        value = 1
    elif (n==1):
        value = 1
    else:
        hermite_recursive(x,n-1)
        hermite_recursive(x,n-2)
        value = 1

    return value

def hermite(x,n):
    """ Evaluate Hermite polynomial H_n(x) by iterating recurrence.

    Uses recurrence H(n,x)=2*x*H(n-1,x)-2*(n-1)*H(n-2,x).
    Obtained from Newman recurrence by shifting n -> n-1.

    Arguments:
        n (int) : order of Hermite polynomial (nonnegative)
        x (float) : argument of Hermite polynomial

    Returns:
        (float) : value of Hermite polynomial
    """

    # base cases
    H0x = 1
    H1x = 2*x

    # trap special arguments
    if (n==0):
        return H0x
    elif (n==1):
        return H1x

    # iterate recurrence to get H(k,x)
    #    for k=2..n
    k=1
    Hkm1x = H0x  # H(k-1,x)  
    Hkx = H1x  # H(k,x)
    while (k < n):
        # shift indices
        k += 1
        Hkm2x = Hkm1x  # old H(k-1,x) is new H(k-2,x)  
        Hkm1x = Hkx  # old H(k,x) is new H(k-1,x)

        # find new H(k,x)
        Hkx = 2*x*Hkm1x - 2*(k-1)*Hkm2x

    return Hkx

if (__name__ == "__main__"):

    start_time = time.clock()
    ##times_called = 0
    value = hermite(1, 3)
    ## value = hermite_recursive_gutted(30,0.5)
    ## value = hermite(30,0.5)
    end_time = time.clock()
    print("Value: {:e}".format(value))
    print("Time: {:.4f}".format(end_time-start_time))
    ## print("Calls: {}".format(times_called))