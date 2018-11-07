'''
------------------------------------------------------------------------
This module contains the functions that generate the variables
associated with firms' optimization in the steady-state or in the
transition path of the overlapping generations model with S-period lived
agents and exogenous labor supply from Chapter 6 of the OG textbook.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_w()
    get_r()
------------------------------------------------------------------------
'''
# Import packages

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_w(K, L, params):
    '''
    --------------------------------------------------------------------
    Solve for steady-state wage w or time path of wages w_t
    --------------------------------------------------------------------
    INPUTS:
    K      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             capital stock or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             labor or time path of aggregate labor
    params = length 2 tuple, (A, alpha)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    A     = scalar > 0, total factor productivity
    alpha = scalar in (0, 1), capital share of income
    w     = scalar > 0 or (T+S-2) vector, steady-state wage or time path
            of wage

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: w
    --------------------------------------------------------------------
    '''
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w


def get_r(K, L, params):
    '''
    --------------------------------------------------------------------
    Solve for steady-state interest rate r or time path of interest
    rates r_t
    --------------------------------------------------------------------
    INPUTS:
    K      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             capital stock or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             labor or time path of aggregate labor
    params = length 3 tuple, (A, alpha, delta)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    A     = scalar > 0, total factor productivity
    alpha = scalar in (0, 1), capital share of income
    delta = scalar in (0, 1), per period depreciation rate
    r     = scalar > 0 or (T+S-2) vector, steady-state interest rate or
            time path of interest rate

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: r
    --------------------------------------------------------------------
    '''
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r


def get_comp_r(r_own, r_other, q, alpha, phi, home):
    '''
    --------------------------------------------------------------------
    This function computes the composite interest rate within a country
    --------------------------------------------------------------------
    '''
    if home:
        q_adj = 1 / q
    else:
        q_adj = q
    r = (((1 - alpha) * (r_own ** (1 - phi))) +
         (alpha * ((q_adj * r_other) ** (1 - phi)))) ** (1 / (1 - phi))

    return r


def get_KLratio(r, gamma, Z, delta):
    '''
    --------------------------------------------------------------------
    This function computes the capital-labor ratio (K/L) given the
    country-specific composite interest rate, gamma, Z, and delta
    --------------------------------------------------------------------
    '''
    KL_rat = (gamma * Z / (r + delta)) ** (1 / (1 - gamma))

    return KL_rat


def get_w_KL(KL_ratio, gamma, Z):
    '''
    --------------------------------------------------------------------
    This function computes the country-specific wage given the country-
    specific capital-labor ratio (K/L), gamma, and Z
    --------------------------------------------------------------------
    '''
    w = (1 - gamma) * Z * (KL_ratio ** gamma)

    return w


def get_K_ownown(K_comp, r_comp, r_own, alpha, phi):
    '''
    --------------------------------------------------------------------
    This function computes the country-specific savings invested in
    own-country intermediate goods production (K_h^h or K_f^f)
    --------------------------------------------------------------------
    '''
    K_ownown = (1 - alpha) * ((r_own / r_comp) ** (-phi)) * K_comp

    return K_ownown


def get_r_own_new(K_other_comp, K_otherown, r_other_comp, q,
                  alpha_other, phi_other, home):
    '''
    --------------------------------------------------------------------
    This function computes the updated value of r_h or r_f
    --------------------------------------------------------------------
    '''
    if home:
        q_adj = 1 / q
    else:
        q_adj = q
    r_own_new = (q_adj * r_other_comp *
                 (alpha_other * K_other_comp / K_otherown) **
                 (1 / phi_other))

    return r_own_new
