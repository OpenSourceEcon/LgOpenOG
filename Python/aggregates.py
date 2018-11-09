'''
------------------------------------------------------------------------
This module contains the functions that generate aggregate variables in
the steady-state or in the transition path of the overlapping
generations model with S-period lived agents and exogenous labor supply
from Chapter 6 of the OG textbook.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_L()
    get_K()
    get_Y()
    get_C()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_L(nvec):
    '''
    --------------------------------------------------------------------
    Solve for aggregate labor L_t
    --------------------------------------------------------------------
    INPUTS:
    nvec = (S,) vector, values for labor supply n_s

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    L = scalar > 0, aggregate labor

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: L
    --------------------------------------------------------------------
    '''
    L = nvec.sum()

    return L


# def get_K(barr):
#     '''
#     --------------------------------------------------------------------
#     Solve for steady-state aggregate capital stock K or time path of
#     aggregate capital stock K_t.

#     We have included a stitching function for K when K<=epsilon such
#     that the the adjusted value is the following. Let sum(b_i) = X.
#     Market clearing is usually given by K = X:

#     K = X when X >= epsilon
#       = f(X) = a * exp(b * X) when X < epsilon
#                    where a = epsilon / e  and b = 1 / epsilon

#     This function has the properties that
#     (i) f(X)>0 and f'(X) > 0 for all X,
#     (ii) f(eps) = eps (i.e., functions X and f(X) meet at epsilon)
#     (iii) f'(eps) = 1 (i.e., slopes of X and f(X) are equal at epsilon)
#     --------------------------------------------------------------------
#     INPUTS:
#     barr = (S,) vector or (S, T+S-2) matrix, values for steady-state
#            savings (b_1, b_2,b_3,...b_S) or time path of the
#            distribution of savings

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

#     OBJECTS CREATED WITHIN FUNCTION:
#     epsilon = scalar > 0, small value at which stitch f(K) function
#     K       = scalar or (T+S-2,) vector, steady-state aggregate capital
#               stock or time path of aggregate capital stock
#     K_cstr  = boolean or (T+S-2) boolean vector, =True if K <= 0 or if
#               K_t <= 0

#     FILES CREATED BY THIS FUNCTION: None

#     RETURNS: K, K_cstr
#     --------------------------------------------------------------------
#     '''
#     epsilon = 0.01
#     a = epsilon / np.exp(1)
#     b = 1 / epsilon
#     if barr.ndim == 1:  # This is the steady-state case
#         K = barr.sum()
#         K_cstr = K < epsilon
#         if K_cstr:
#             print('get_K() warning: Distribution of savings and/or ' +
#                   'parameters created K < epsilon')
#             # Force K >= eps by stitching a * exp(b * K) for K < eps
#             K = a * np.exp(b * K)

#     elif barr.ndim == 2:  # This is the time path case
#         K = barr.sum(axis=0)
#         K_cstr = K < epsilon
#         if K.min() < epsilon:
#             print('get_K() warning: Aggregate capital constraint is ' +
#                   'violated (K < epsilon) for some period in time ' +
#                   'path.')
#             K[K_cstr] = a * np.exp(b * K[K_cstr])

#     return K, K_cstr


def get_Y(K, L, gamma, Z):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate output Y or time path of aggregate
    output Y_t
    --------------------------------------------------------------------
    INPUTS:
    K      = scalar > 0 or (T+S-2,) vector, aggregate capital stock
             or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, aggregate labor or time
             path of the aggregate labor
    params = length 2 tuple, production function parameters
             (A, alpha)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    A     = scalar > 0, total factor productivity
    alpha = scalar in (0,1), capital share of income
    Y     = scalar > 0 or (T+S-2,) vector, aggregate output (GDP) or
            time path of aggregate output (GDP)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Y
    --------------------------------------------------------------------
    '''
    Y = Z * (K ** gamma) * (L ** (1 - gamma))

    return Y


def get_C(carr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate consumption C or time path of
    aggregate consumption C_t
    --------------------------------------------------------------------
    INPUTS:
    carr = (S,) vector or (S, T) matrix, distribution of consumption c_s
           in steady state or time path for the distribution of
           consumption

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    C = scalar > 0 or (T,) vector, aggregate consumption or time path of
        aggregate consumption

    Returns: C
    --------------------------------------------------------------------
    '''
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C


def get_I(K_t, K_tp1, delta):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate investment I or time path of
    aggregate Investment I_t in Home or Foreign country
    --------------------------------------------------------------------
    INPUTS:
    K_t   = scalar > 0 or (T,) vector, steady-state or time path of
            current period aggregate final goods capital in Home or
            Foreign country
    K_tp1 = scalar > 0 or (T,) vector, steady-state or time path of next
            period aggregate final goods capital in Home or Foreign
            country
    delta = scalar in [0, 1], model-period depreciation rate of final
            goods capital

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    I_t = scalar > 0 or (T,) vector, steady-state aggregate investment
          or time path of aggregate consumption

    Returns: I_t
    --------------------------------------------------------------------
    '''
    I_t = K_tp1 - (1 - delta) * K_t

    return I_t


def get_NX(K_ownown_t, K_othown_t, K_ownown_tp1, K_othown_tp1, K_t,
           K_tp1, r_t, r_own_t):
    '''
    --------------------------------------------------------------------
    Solve for steady-state net exports NX or time path of net exports
    NX_t in Home or Foreign country
    --------------------------------------------------------------------
    INPUTS:
    K_t     = scalar > 0 or (T,) vector, steady-state or time path of
              current period aggregate final goods capital in Home or
              Foreign country
    K_tp1   = scalar > 0 or (T,) vector, steady-state or time path of
              next period aggregate final goods capital in Home or
              Foreign country
    r_t     =
    r_own_t =

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    I_t = scalar > 0 or (T,) vector, steady-state aggregate investment
          or time path of aggregate consumption

    Returns: I_t
    --------------------------------------------------------------------
    '''
    NX_t = (K_ownown_tp1 + K_othown_tp1 -
            (1 + r_own_t) * (K_ownown_t + K_othown_t) -
            (K_tp1 - (1 + r_t) * K_t))

    return NX_t
