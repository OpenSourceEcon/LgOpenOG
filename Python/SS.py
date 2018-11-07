'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents and exogenous labor supply from
Chapter 6 of the OG textbook.

This Python module imports the following module(s):
    households.py
    firms.py
    aggregates.py
    utilities.py

This Python module defines the following function(s):
    feasible()
    inner_loop()
    rw_errs()
    KL_errs()
    get_SS_root()
    get_SS_bsct()
    create_graphs()
------------------------------------------------------------------------
'''
# Import packages
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import household as hh
import firms
import aggregates as aggr
import utilities as utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def feasible(bvec, params):
    '''
    --------------------------------------------------------------------
    Check whether a vector of steady-state savings is feasible in that
    it satisfies the nonnegativity constraints on consumption in every
    period c_s > 0 and that the aggregate capital stock is strictly
    positive K > 0
    --------------------------------------------------------------------
    INPUTS:
    bvec   = (S-1,) vector, household savings b_{s+1}
    params = length 4 tuple, (nvec, A, alpha, delta)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_L()
        get_K()
        get_w()
        get_r()
        get_cvec()

    OBJECTS CREATED WITHIN FUNCTION:
    nvec     = (S,) vector, exogenous labor supply values n_s
    A        = scalar > 0, total factor productivity
    alpha    = scalar in (0, 1), capital share of income
    delta    = scalar in (0, 1), per-period depreciation rate
    S        = integer >= 3, number of periods in individual life
    L        = scalar > 0, aggregate labor
    K        = scalar, steady-state aggregate capital stock
    K_cstr   = boolean, =True if K <= 0
    w_params = length 2 tuple, (A, alpha)
    w        = scalar, steady-state wage
    r_params = length 3 tuple, (A, alpha, delta)
    r        = scalar, steady-state interest rate
    bvec2    = (S,) vector, steady-state savings distribution plus
               initial period wealth of zero
    cvec     = (S,) vector, steady-state consumption by age
    c_cnstr  = (S,) Boolean vector, =True for elements for which c_s<=0
    b_cnstr  = (S-1,) Boolean, =True for elements for which b_s causes a
               violation of the nonnegative consumption constraint

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_cnstr, c_cnstr, K_cnstr
    --------------------------------------------------------------------
    '''
    nvec, A, alpha, delta = params
    S = nvec.shape[0]
    L = aggr.get_L(nvec)
    K, K_cstr = aggr.get_K(bvec)
    if not K_cstr:
        w_params = (A, alpha)
        w = firms.get_w(K, L, w_params)
        r_params = (A, alpha, delta)
        r = firms.get_r(K, L, r_params)
        c_params = (nvec, r, w)
        cvec = hh.get_cons(bvec, 0.0, c_params)
        c_cstr = cvec <= 0
        b_cstr = c_cstr[:-1] + c_cstr[1:]

    else:
        c_cstr = np.ones(S, dtype=bool)
        b_cstr = np.ones(S - 1, dtype=bool)

    return c_cstr, K_cstr, b_cstr


def get_ss_graphs(c_ss, b_ss, home):
    '''
    --------------------------------------------------------------------
    Plot steady-state results
    --------------------------------------------------------------------
    INPUTS:
    c_ss = (S,) vector, steady-state lifetime consumption
    b_ss = (S-1,) vector, steady-state lifetime savings

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    S           = integer > 3, number of periods in lifetime
    bss2        = (S,) vector, [0, b_ss]
    age_pers    = (S,) vector, ages from 1 to S

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png

    RETURNS: None
    ----------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    # Plot steady-state consumption and savings distributions
    S = c_ss.shape[0]
    b_ss2 = np.append(0.0, b_ss)
    age_pers = np.arange(1, S + 1)
    fig, ax = plt.subplots()
    plt.plot(age_pers, c_ss, marker='D', label='Consumption')
    plt.plot(age_pers, b_ss2, marker='D', label='Savings')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Steady-state consumption and savings', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Units of consumption')
    plt.xlim((0, S + 1))
    # plt.ylim((-1.0, 1.15 * (b_ss.max())))
    plt.legend(loc='upper left')
    if home:
        filename = 'SS_bc_h'
    else:
        filename = 'SS_bc_f'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    # plt.show()
    plt.close()


def outer_loop(rh_init, rf_init, q_init, args):
    '''
    --------------------------------------------------------------------
    Bisection method solving for rh, rf, and q that solves for the outer-loop
    --------------------------------------------------------------------
    '''
    (nhvec, nfvec, beta, sigma, alpha_h, phi_h, Z_h, gamma_h, delta_h,
        alpha_f, phi_f, Z_f, gamma_f, delta_f, tol_outer, tol_inner, xi,
        maxiter, EulDiff) = args

    S = nhvec.shape[0]
    dist = 10.0
    SS_iter = 0
    rhrfq_init = np.array([rh_init, rf_init, q_init])
    print('SS iter=', SS_iter, ', Dist=', dist, ', (r_h, r_f, q)=',
          rhrfq_init)
    while dist >= tol_outer and SS_iter <= maxiter:
        SS_iter += 1
        # Solve for composite r and rstar
        rh_comp = firms.get_comp_r(rh_init, rf_init, q_init, alpha_h,
                                   phi_h, home=True)
        rf_comp = firms.get_comp_r(rf_init, rh_init, q_init, alpha_f,
                                   phi_f, home=False)
        # Solve for capital-labor ratios and wages in both countries
        KL_rat_h = firms.get_KLratio(rh_comp, gamma_h, Z_h, delta_h)
        KL_rat_f = firms.get_KLratio(rf_comp, gamma_f, Z_f, delta_f)
        w_h = firms.get_w_KL(KL_rat_h, gamma_h, Z_h)
        w_f = firms.get_w_KL(KL_rat_f, gamma_f, Z_f)
        # Solve for household decisions in Home country
        r_h_path = rh_init * np.ones(S)
        w_h_path = w_h * np.ones(S)
        bhvec_init = 0.02 * np.ones(S - 1)
        cbh_args = (0.0, nhvec, beta, sigma, EulDiff, tol_inner)
        bhvec, chvec, bh_errors, success_h = \
            hh.get_cbvec(bhvec_init, r_h_path, w_h_path, cbh_args)
        # Solve for household decisions in Foreign country
        r_f_path = rf_init * np.ones(S)
        w_f_path = w_f * np.ones(S)
        bfvec_init = 0.02 * np.ones(S - 1)
        cbf_args = (0.0, nfvec, beta, sigma, EulDiff, tol_inner)
        bfvec, cfvec, bf_errors, success_f = \
            hh.get_cbvec(bfvec_init, r_f_path, w_f_path, cbf_args)
        # Solve for Home and Foreign aggregate labor supply
        L_h = aggr.get_L(nhvec)
        L_f = aggr.get_L(nfvec)
        # Solve for Home and Foreign aggregate final goods capital stock
        K_h = KL_rat_h * L_h
        K_f = KL_rat_f * L_f
        # Solve for Home and Foreign savings invested in own-country
        # intermediate goods production
        K_hh = firms.get_K_ownown(K_h, rh_comp, rh_init, alpha_h, phi_h)
        K_ff = firms.get_K_ownown(K_f, rf_comp, rf_init, alpha_f, phi_f)
        # Solve for Home and Foreign savings invested in other-country
        # intermediate goods production
        K_fh = bhvec.sum() - K_hh
        K_hf = bfvec.sum() - K_ff
        # Compute new values for r_h, r_f, and q and calulate distance
        rh_new = firms.get_r_own_new(K_f, K_fh, rf_comp, q_init,
                                     alpha_f, phi_f, home=True)
        rf_new = firms.get_r_own_new(K_h, K_hf, rh_comp, q_init,
                                     alpha_h, phi_h, home=False)
        q_new = (rf_init * K_hf) / (rh_init * K_fh)
        rhrfq_new = np.array([rh_new, rf_new, q_new])
        dist = ((rhrfq_new - rhrfq_init) ** 2).sum()
        # Update initial values of outer loop variables
        rhrfq_init = xi * rhrfq_new + (1 - xi) * rhrfq_init
        print('SS iter=', SS_iter, ', Dist=', dist, ', (r_h, r_f, q)=',
              rhrfq_init)

    if dist >= tol_outer:
        success = False
    else:
        success = True
    rh_ss, rf_ss, q_ss = rhrfq_init

    return rh_ss, rf_ss, q_ss, dist, success


def get_SS(rh_ss_guess, rf_ss_guess, q_ss_guess, args, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with exogenous labor supply using one root finder in bvec
    --------------------------------------------------------------------
    INPUTS:
    bss_guess = (S-1,) vector, initial guess for b_ss
    args      = length 8 tuple,
                (nvec, beta, sigma, A, alpha, delta, SS_tol, SS_EulDiff)
    graphs    = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        SS_EulErrs()
        aggr.get_K()
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        firms.get_r()
        firms.get_w()
        hh.get_cons()
        utils.print_time()
        get_ss_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time = scalar > 0, clock time at beginning of program
    nvec       = (S,) vector, exogenous lifetime labor supply n_s
    beta       = scalar in (0,1), discount factor for each model per
    sigma      = scalar > 0, coefficient of relative risk aversion
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    alpha      = scalar in (0,1), capital share of income
    delta      = scalar in [0,1], model-period depreciation rate of
                 capital
    SS_tol     = scalar > 0, tolerance level for steady-state fsolve
    SS_EulDiff = Boolean, =True if want difference version of Euler
                 errors beta*(1+r)*u'(c2) - u'(c1), =False if want ratio
                 version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    b_args     = length 7 tuple, args passed to opt.root(SS_EulErrs,...)
    results_b  = results object, output from opt.root(SS_EulErrs,...)
    b_ss       = (S-1,) vector, steady-state savings b_{s+1}
    K_ss       = scalar > 0, steady-state aggregate capital stock
    Kss_cstr   = boolean, =True if K_ss < epsilon
    L          = scalar > 0, exogenous aggregate labor
    r_params   = length 3 tuple, (A, alpha, delta)
    r_ss       = scalar > 0, steady-state interest rate
    w_params   = length 2 tuple, (A, alpha)
    w_ss       = scalar > 0, steady-state wage
    c_args     = length 3 tuple, (nvec, r_ss, w_ss)
    c_ss       = (S,) vector, steady-state individual consumption c_s
    Y_params   = length 2 tuple, (A, alpha)
    Y_ss       = scalar > 0, steady-state aggregate output (GDP)
    C_ss       = scalar > 0, steady-state aggregate consumption
    b_err_ss   = (S-1,) vector, Euler errors associated with b_ss
    RCerr_ss   = scalar, steady-state resource constraint error
    ss_time    = scalar > 0, time elapsed during SS computation
                 (in seconds)
    ss_output  = length 10 dict, steady-state objects {b_ss, c_ss, w_ss,
                 r_ss, K_ss, Y_ss, C_ss, b_err_ss, RCerr_ss, ss_time}

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    (nhvec, nfvec, beta, sigma, alpha_h, phi_h, Z_h, gamma_h, delta_h,
        alpha_f, phi_f, Z_f, gamma_f, delta_f, tol_outer, tol_inner, xi,
        maxiter, EulDiff) = args
    S = nhvec.shape[0]
    results_ol = outer_loop(rh_ss_guess, rf_ss_guess, q_ss_guess, args)
    rh_ss, rf_ss, q_ss, ol_dist, ol_success = results_ol
    # Solve for composite r and rstar
    r_ss = firms.get_comp_r(rh_ss, rf_ss, q_ss, alpha_h, phi_h,
                            home=True)
    rstar_ss = firms.get_comp_r(rf_ss, rh_ss, q_ss, alpha_f, phi_f,
                                home=False)
    # Solve for capital-labor ratios and wages in both countries
    KL_rat_h = firms.get_KLratio(r_ss, gamma_h, Z_h, delta_h)
    KL_rat_f = firms.get_KLratio(rstar_ss, gamma_f, Z_f, delta_f)
    wh_ss = firms.get_w_KL(KL_rat_h, gamma_h, Z_h)
    wf_ss = firms.get_w_KL(KL_rat_f, gamma_f, Z_f)
    # Solve for household decisions in Home country
    r_h_path = rh_ss * np.ones(S)
    w_h_path = wh_ss * np.ones(S)
    bhvec_init = 0.02 * np.ones(S - 1)
    cbh_args = (0.0, nhvec, beta, sigma, EulDiff, tol_inner)
    bh_ss, ch_ss, bhss_errors, success_h = \
        hh.get_cbvec(bhvec_init, r_h_path, w_h_path, cbh_args)
    # Solve for household decisions in Foreign country
    r_f_path = rf_ss * np.ones(S)
    w_f_path = wf_ss * np.ones(S)
    bfvec_init = 0.02 * np.ones(S - 1)
    cbf_args = (0.0, nfvec, beta, sigma, EulDiff, tol_inner)
    bf_ss, cf_ss, bfss_errors, success_f = \
        hh.get_cbvec(bfvec_init, r_f_path, w_f_path, cbf_args)
    # Solve for Home and Foreign aggregate labor supply
    L_h_ss = aggr.get_L(nhvec)
    L_f_ss = aggr.get_L(nfvec)
    # Solve for Home and Foreign aggregate final goods capital stock
    K_h_ss = KL_rat_h * L_h_ss
    K_f_ss = KL_rat_f * L_f_ss
    # Solve for Home and Foreign aggregate final goods output (GDP)
    Yh_ss = aggr.get_Y(K_h_ss, L_h_ss, gamma_h, Z_h)
    Yf_ss = aggr.get_Y(K_f_ss, L_f_ss, gamma_f, Z_f)
    # Solve for Home and Foreign savings invested in own-country
    # intermediate goods production
    K_hh_ss = firms.get_K_ownown(K_h_ss, r_ss, rh_ss, alpha_h, phi_h)
    K_ff_ss = firms.get_K_ownown(K_f_ss, rstar_ss, rf_ss, alpha_f,
                                 phi_f)
    # Solve for Home and Foreign savings invested in other-country
    # intermediate goods production
    K_fh_ss = bh_ss.sum() - K_hh_ss
    K_hf_ss = bf_ss.sum() - K_ff_ss
    # Solve for Home and Foreign aggregate consumption
    Ch_ss = ch_ss.sum()
    Cf_ss = cf_ss.sum()
    # Solve for Home and Foreign goods market clearing (resource
    # constraint) errors
    # RC_h_err_ss = Yh_ss - Ch_ss - delta_h * K_h_ss
    # RC_f_err_ss = Yf_ss - Cf_ss - delta_f * K_f_ss
    RC_h_err_ss = (Yh_ss - Ch_ss - (r_ss + delta_h) * K_h_ss +
                   rh_ss * (K_hh_ss + K_fh_ss))
    RC_f_err_ss = (Yf_ss - Cf_ss - (rstar_ss + delta_f) * K_f_ss +
                   rf_ss * (K_ff_ss + K_hf_ss))

    ss_time = time.clock() - start_time

    ss_output = \
        {'bh_ss': bh_ss, 'ch_ss': ch_ss, 'bhss_errors': bhss_errors,
         'bf_ss': bf_ss, 'cf_ss': cf_ss, 'bfss_errors': bfss_errors,
         'wh_ss': wh_ss, 'rh_ss': rh_ss, 'r_ss': r_ss, 'q_ss': q_ss,
         'wf_ss': wf_ss, 'rf_ss': rf_ss, 'rstar_ss': rstar_ss,
         'L_h_ss': L_h_ss, 'K_h_ss': K_h_ss, 'K_hh_ss': K_hh_ss,
         'K_fh_ss': K_fh_ss, 'Yh_ss': Yh_ss, 'L_f_ss': L_f_ss,
         'K_f_ss': K_f_ss, 'K_ff_ss': K_ff_ss, 'K_hf_ss': K_hf_ss,
         'Yf_ss': Yf_ss, 'RC_h_err_ss': RC_h_err_ss,
         'RC_f_err_ss': RC_f_err_ss, 'ss_time': ss_time}

    print('bh_ss is: ', bh_ss)
    print('bf_ss is: ', bf_ss)
    print('K_h_ss=', K_h_ss, ', rh_ss=', rh_ss, ', r_ss=', r_ss,
          ', wh_ss=', wh_ss)
    print('K_f_ss=', K_f_ss, ', rf_ss=', rf_ss, ', rstar_ss=', rstar_ss,
          ', wf_ss=', wf_ss)
    print('Max. abs. savings Euler error is: ',
          np.absolute(np.append(bhss_errors, bfss_errors)).max())
    print('Max. abs. resource constraint error is: ',
          np.absolute(np.append(RC_h_err_ss, RC_f_err_ss)).max())

    # Print SS computation time
    utils.print_time(ss_time, 'SS')

    if graphs:
        get_ss_graphs(ch_ss, bh_ss, home=True)
        get_ss_graphs(cf_ss, bf_ss, home=False)

    return ss_output
