'''
------------------------------------------------------------------------
This module contains the functions that generate the variables
associated with households' optimization in the steady-state or in the
transition path of the two-country overlapping generations model with S-
period lived agents and exogenous labor supply from the large open
economy chapter of the OG textbook.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    feasible()
    get_cons()
    MU_c_stitch()
    get_b_errors()
                            bn_solve()
                            FOC_savings()
                            FOC_labor()
                            get_cnb_vecs()
                            c1_bSp1err()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def c_feasible(b_s, b_sp1, args):
    '''
    '''
    c_s = get_cons(b_s, b_sp1, args)
    c_gt0 = c_s > 0
    c_lteq0 = 1 - c_gt0
    if c_lteq0.sum() > 0:
        feas_bool = False
        print('feasible(): Initial guess for bvec is not feasible.')
        print('Elements of resulting c_s that are <= 0:')
        print(c_lteq0)
    feas_bool = True

    return feas_bool, c_lteq0


def get_cons(b_s, b_sp1, args):
    '''
    --------------------------------------------------------------------
    Calculate household consumption given prices, labor supply, current
    wealth, and savings
    --------------------------------------------------------------------
    INPUTS:
    b_s   = scalar or (p,) vector, initial wealth in all p remaining
            periods of life
    b_sp1 = scalar or (p,) vector, savings in all p remaining periods of
            life
    args  = length 3 tuple, (nvec, r, w)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    n_s = scalar >= or (p,) vector, labor supply over remaining life
    r   = scalar > 0 or (p,) vector, interest rate over remaining life
    w   = scalar > 0 or (p,) vector, wage over remaining life
    c_s = scalar > 0 or (p,) vector, consumption over remaining life

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: c_s
    --------------------------------------------------------------------
    '''
    n_s, r, w = args
    c_s = ((1 + r) * b_s) + (w * n_s) - b_sp1

    return c_s


def MU_c_stitch(cvec, sigma, graph=False):
    '''
    --------------------------------------------------------------------
    Generate marginal utility(ies) of consumption with CRRA consumption
    utility and stitched function at lower bound such that the new
    hybrid function is defined over all consumption on the real
    line but the function has similar properties to the Inada condition.

    u'(c) = c ** (-sigma) if c >= epsilon
          = g'(c) = 2 * b2 * c + b1 if c < epsilon

        such that g'(epsilon) = u'(epsilon)
        and g''(epsilon) = u''(epsilon)

        u(c) = (c ** (1 - sigma) - 1) / (1 - sigma)
        g(c) = b2 * (c ** 2) + b1 * c + b0
    --------------------------------------------------------------------
    INPUTS:
    cvec  = scalar or (p,) vector, individual consumption value or
            lifetime consumption over p consecutive periods
    sigma = scalar >= 1, coefficient of relative risk aversion for CRRA
            utility function: (c**(1-sigma) - 1) / (1 - sigma)
    graph = boolean, =True if want plot of stitched marginal utility of
            consumption function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, positive value close to zero
    c_s        = scalar, individual consumption
    c_s_cnstr  = boolean, =True if c_s < epsilon
    b1         = scalar, intercept value in linear marginal utility
    b2         = scalar, slope coefficient in linear marginal utility
    MU_c       = scalar or (p,) vector, marginal utility of consumption
                 or vector of marginal utilities of consumption
    p          = integer >= 1, number of periods remaining in lifetime
    cvec_cnstr = (p,) boolean vector, =True for values of cvec < epsilon

    FILES CREATED BY THIS FUNCTION:
        MU_c_stitched.png

    RETURNS: MU_c
    --------------------------------------------------------------------
    '''
    epsilon = 0.0001
    if np.ndim(cvec) == 0:
        c_s = cvec
        c_s_cnstr = c_s < epsilon
        if c_s_cnstr:
            b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
            b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
            MU_c = 2 * b2 * c_s + b1
        else:
            MU_c = c_s ** (-sigma)
    elif np.ndim(cvec) == 1:
        p = cvec.shape[0]
        cvec_cnstr = cvec < epsilon
        MU_c = np.zeros(p)
        MU_c[~cvec_cnstr] = cvec[~cvec_cnstr] ** (-sigma)
        b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
        b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
        MU_c[cvec_cnstr] = 2 * b2 * cvec[cvec_cnstr] + b1

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        cvec_CRRA   = (1000,) vector, support of c including values
                      between 0 and epsilon
        MU_CRRA     = (1000,) vector, CRRA marginal utility of
                      consumption
        cvec_stitch = (500,) vector, stitched support of consumption
                      including negative values up to epsilon
        MU_stitch   = (500,) vector, stitched marginal utility of
                      consumption
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        cvec_CRRA = np.linspace(epsilon / 2, epsilon * 3, 1000)
        MU_CRRA = cvec_CRRA ** (-sigma)
        cvec_stitch = np.linspace(-0.00005, epsilon, 500)
        MU_stitch = 2 * b2 * cvec_stitch + b1
        fig, ax = plt.subplots()
        plt.plot(cvec_CRRA, MU_CRRA, ls='solid', label='$u\'(c)$: CRRA')
        plt.plot(cvec_stitch, MU_stitch, ls='dashed', color='red',
                 label='$g\'(c)$: stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched ' +
                  'function', fontsize=14)
        plt.xlabel(r'Consumption $c$')
        plt.ylabel(r'Marginal utility $u\'(c)$')
        plt.xlim((-0.00005, epsilon * 3))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "MU_c_stitched")
        plt.savefig(output_path)
        # plt.show()

    return MU_c


def b_errors(c_vec, args):
    '''
    --------------------------------------------------------------------
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings decision. Because this function is used for
    solving for lifetime decisions in both the steady-state and in the
    transition path, lifetimes will be of varying length. Lifetimes in
    the steady-state will be S periods. Lifetimes in the transition path
    will be p in [2, S] periods
    --------------------------------------------------------------------
    INPUTS:
    c_vec = (p,) vector, consumption over p remaining life periods, p>=2
            periods
    args  = length 4 tuple, (beta, sigma, r, simp_diff)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        MU_c_stitch()

    OBJECTS CREATED WITHIN FUNCTION:
    beta      = scalar in (0,1), discount factor
    sigma     = scalar > 0, coefficient of relative risk aversion
    r         = scalar > 0 or (p-1,) vector, steady-state interest rate
                or time path of interest rates
    simp_diff = boolean, =True if use simple difference Euler errors.
                Use percent difference errors otherwise.
    mu_c      = scalar or (p-1,) vector, marginal utility of current
                consumption
    mu_cp1    = scalar or (p-1,) vector, marginal utility of next period
                consumption
    b_errors = scalar or (p-1,) vector, Euler error(s) characterizing
               optimal savings bvec

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
    beta, sigma, r, simp_diff = args
    if c_vec.shape[0] == 2:  # Make MU's scalars in this case
        mu_c = MU_c_stitch(c_vec[0], sigma)
        mu_cp1 = MU_c_stitch(c_vec[1], sigma)
    elif c_vec.shape[0] > 2:
        mu_c = MU_c_stitch(c_vec[:-1], sigma)
        mu_cp1 = MU_c_stitch(c_vec[1:], sigma)

    if not np.isscalar(r) and r.shape[0] == 2:
        r = r[-1]
    elif not np.isscalar(r) and r.shape[0] > 2:
        r = r[1:]

    if simp_diff:
        b_errors = (beta * (1 + r) * mu_cp1) - mu_c
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1

    return b_errors


def get_b_errors(bvec, *args):
    b_cur, r, w, nvec, beta, sigma, simp_diff = args
    b_s = np.append(b_cur, bvec)
    b_sp1 = np.append(bvec, 0.0)
    c_args = (nvec, r, w)
    cvec = get_cons(b_s, b_sp1, c_args)
    b_args = (beta, sigma, r, simp_diff)
    b_errs = b_errors(cvec, b_args)

    return b_errs


def get_cbvec(bvec_init, rpath, wpath, cb_args):
    '''
    --------------------------------------------------------------------
    This function solves for the remaining lifetime savings decisions of
    a household and returns the optimal savings vector (p-1) and
    corresponding consumption vector (p)
    --------------------------------------------------------------------
    INPUTS:
    bvec_init = (p-1,) vector, initial guess for bvec
    rpath     = (p,) vector, time path of interest rates over p
                remaining periods
    wpath     = (p,) vector, time path of wages over p remaining periods
    cb_args   = length 5 tuple, (b_cur, nvec, beta, sigma, simp_diff)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        MU_c_stitch()

    OBJECTS CREATED WITHIN FUNCTION:
    beta      = scalar in (0,1), discount factor
    sigma     = scalar > 0, coefficient of relative risk aversion
    r         = scalar > 0 or (p-1,) vector, steady-state interest rate
                or time path of interest rates
    simp_diff = boolean, =True if use simple difference Euler errors.
                Use percent difference errors otherwise.
    mu_c      = scalar or (p-1,) vector, marginal utility of current
                consumption
    mu_cp1    = scalar or (p-1,) vector, marginal utility of next period
                consumption
    b_errors = scalar or (p-1,) vector, Euler error(s) characterizing
               optimal savings bvec

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
    b_cur, nvec, beta, sigma, simp_diff, tol_inner = cb_args
    c_args = (nvec, rpath, wpath)
    if np.isscalar(nvec):
        bvec = 0.0
        cvec = get_cons(b_cur, bvec, c_args)
    else:
        feas_bool = False
        f_args = (nvec, rpath, wpath)
        while not feas_bool:
            feas_bool, c_lteq0 = c_feasible(np.append(b_cur, bvec_init),
                                            np.append(bvec_init, 0.0),
                                            f_args)
            if not feas_bool:
                bvec_init *= 0.75
        b_args = (b_cur, rpath, wpath, nvec, beta, sigma, simp_diff)
        results_b = opt.root(get_b_errors, bvec_init, args=(b_args),
                             tol=tol_inner)
        bvec = results_b.x
        b_errors = results_b.fun
        success = results_b.success
        b_s = np.append(b_cur, bvec)
        b_sp1 = np.append(bvec, 0.0)
        cvec = get_cons(b_s, b_sp1, c_args)

    return bvec, cvec, b_errors, success
