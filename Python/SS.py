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
    get_ss_graphs()
    outer_loop()
    get_SS()
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


def get_ss_graphs(c_ss, b_ss, home):
    '''
    --------------------------------------------------------------------
    Plot steady-state household results for Home or Foreign country
    --------------------------------------------------------------------
    INPUTS:
    c_ss = (S,) vector, steady-state lifetime consumption
    b_ss = (S-1,) vector, steady-state lifetime savings
    home = boolean, =True if plot is for Home country. Otherwise, the
           plot is for the Foreign country

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    S           = integer > 3, number of periods in lifetime
    bss2        = (S,) vector, [0, b_ss]
    age_pers    = (S,) vector, ages from 1 to S

    FILES CREATED BY THIS FUNCTION:
        SS_bc_h.png or SS_bc_f.png

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
    Bisection method solving for rh, rf, and q that solves for the
    outer-loop
    --------------------------------------------------------------------
    INPUTS:
    rh_init = scalar > 0, initial guess for rh_ss
    rf_init = scalar > 0, initial guess for rf_ss
    q_init  = scalar > 0, initial guess for q_ss
    args    = length 19 tuple, arguments for function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        firms.get_comp_r()
        firms.get_KLratio()
        firms.get_w_KL()
        hh.get_cbvec()
        aggr.get_L()
        firms.get_K_ownown()
        firms.get_r_own_new()

    OBJECTS CREATED WITHIN FUNCTION:
    S          = integer in [3,80], number of periods an individual
                 lives
    nhvec      = (S,) vector, exogenous Home labor supply n_{h,s,t}
    nfvec      = (S,) vector, exogenous Foreign labor supply n_{h,s,t}
    beta       = scalar in (0,1), discount factor for each model period
    sigma      = scalar > 0, coefficient of relative risk aversion
    alpha_h    = scalar in (0, 1), share parameter in CES production
                 function of Home intermediate goods producer
    phi_h      = scalar >= 1, elasticity of substitution in CES
                 production function of Home intermediate goods producer
    Z_h        = scalar > 0, total factor productivity parameter in Home
                 final goods producer production function
    gamma_h    = scalar in (0, 1), capital share of income in Home Cobb-
                 Douglas production function
    delta_h    = scalar in [0,1], model-period depreciation rate of Home
                 final goods capital
    alpha_f    = scalar in (0, 1), share parameter in CES production
                 function of Foreign intermediate goods producer
    phi_f      = scalar >= 1, elasticity of substitution in CES
                 production function of Foreign intermediate goods
                 producer
    Z_f        = scalar > 0, total factor productivity parameter in
                 Foreign final goods producer production function
    gamma_f    = scalar in (0, 1), capital share of income in Foreign
                 Cobb-Douglas production function
    delta_f    = scalar in [0,1], model-period depreciation rate of
                 Foreign final goods capital
    tol_outer  = scalar > 0, tolerance level for steady-state outer-loop
                 convergence
    tol_inner  = scalar > 0, tolerance level for inner-loop root finder
    xi         = scalar in (0, 1], outer loop updating parameter
    maxiter    = integer >= 1, maximum number of iterations for outer
                 loop fixed point algorithm
    EulDiff    = boolean, =True if use simple differences in Euler
                 errors. Otherwise, use percent deviation form
    dist       = scalar > 0, distance measure between initial guess and
                 predicted value
    SS_iter    = integer >= 0, iteration number of fixed point algorithm
    rhrfq_init = (3,) vector, initial values of (rh, rf, q)
    rh_comp    = scalar > -delta_h, Home composite consumption
    rf_comp    = scalar > -delta_f, Foreign composite consumption
    KL_rat_h   = scalar > 0, Home final goods capital labor ratio
    KL_rat_f   = scalar > 0, Foreign final goods capital labor ratio
    w_h        = scalar > 0, steady-state Home wage
    w_f        = scalar > 0, steady-state Foreign wage
    r_h_path   = (S,) vector, steady-state Home composite interest rate
                 over the periods of an agent's life
    w_h_path   = (S,) vector, steady-state Home wage over the periods
                 of an agent's life
    bhvec_init = (S-1,) vector, initial guess for Home steady-state
                 lifetime savings vector
    cbh_args   = length 6 tuple, arguments to pass in to hh.get_cbvec()
    bhvec      = (S-1,) vector, steady-state Home household savings for
                 each age (b_2, b_3, ...b_S)
    chvec      = (S,) vector, steady-state Home household consumption
                 for each age (c_1, c_2, ...c_S)
    bh_errors  = (S-1,) vector, steady-state Home household Euler
                 equation errors for savings decision
    success_h  = boolean, =True if root finder for inner loop Home
                 household savings decision converged
    r_f_path   = (S,) vector, steady-state Foreign composite interest
                 rate over the periods of an agent's life
    w_f_path   = (S,) vector, steady-state Foreign wage over the periods
                 of an agent's life
    bfvec_init = (S-1,) vector, initial guess for Foreign steady-state
                 lifetime savings vector
    cbf_args   = length 6 tuple, arguments to pass in to hh.get_cbvec()
    bfvec      = (S-1,) vector, steady-state Foreign household savings
                 for each age (b_2, b_3, ...b_S)
    cfvec      = (S,) vector, steady-state Foreign household consumption
                 for each age (c_1, c_2, ...c_S)
    bf_errors  = (S-1,) vector, steady-state Foreign household Euler
                 equation errors for savings decision
    success_f  = boolean, =True if root finder for inner loop Foreign
                 household savings decision converged
    L_h        = scalar > 0, steady-state Home aggregate labor
    L_f        = scalar > 0, steady-state Foreign aggregate labor
    K_h        = scalar > 0, steady-state Home final goods capital stock
    K_f        = scalar > 0, steady-state Foreign final goods capital
                 stock
    K_hh       = scalar > 0, steady-state total Home country savings
                 allocated to Home country intermed't goods production
    K_ff       = scalar > 0, steady-state total Foreign country savings
                 allocated to Foreign country int'd't goods production
    K_fh       = scalar > 0, steady-state total Home country savings
                 allocated to Foreign country int'd't goods production
    K_hf       = scalar > 0, steady-state total Foreign country savings
                 allocated to Home country intermed't goods production
    rh_new     = scalar > 0, new predicted value for rh_ss
    rf_new     = scalar > 0, new predicted value for rf_ss
    q_new      = scalar > 0, new predicted value for q_ss
    rhrfq_new  = (3,) vector, new values of (rh, rf, q)
    success    = boolean, =True if outer loop algorithm converged
    rh_ss      = scalar > 0, steady-state return on Home savings
    rf_ss      = scalar > 0, steady-state return on Foreign savings
    q_ss       = scalar > 0, real exchange rate # Foreign consumption
                 goods per 1 Domestic consumption good

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: rh_ss, rf_ss, q_ss, dist, success
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
        print('SS iter=', SS_iter, ', Dist=', '%10.4e' % (dist))
        # print(', (r_h, r_f, q)=', rhrfq_init)

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
    rh_ss_guess = scalar > 0, initial guess for rh_ss
    rf_ss_guess = scalar > 0, initial guess for rf_ss
    q_ss_guess  = scalar > 0, initial guess for q_ss
    args        = length 19 tuple, arguments passed in to get_SS()
    graphs      = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        outer_loop()
        firms.get_comp_r()
        firms.get_KLratio()
        firms.get_w_KL()
        hh.get_cbvec()
        aggr.get_L()
        aggr.get_Y()
        firms.get_K_ownown()
        aggr.get_C()
        aggr.get_I()
        aggr.get_NX()
        utils.print_time()
        get_ss_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time  = scalar > 0, clock time at beginning of program
    S           = integer in [3,80], number of periods an individual
                  lives
    nhvec       = (S,) vector, exogenous Home labor supply n_{h,s,t}
    nfvec       = (S,) vector, exogenous Foreign labor supply n_{h,s,t}
    beta        = scalar in (0,1), discount factor for each model period
    sigma       = scalar > 0, coefficient of relative risk aversion
    alpha_h     = scalar in (0, 1), share parameter in CES production
                  function of Home intermediate goods producer
    phi_h       = scalar >= 1, elasticity of substitution in CES
                  production function of Home intermediate goods
                  producer
    Z_h         = scalar > 0, total factor productivity parameter in
                  Home final goods producer production function
    gamma_h     = scalar in (0, 1), capital share of income in Home
                  Cobb-Douglas production function
    delta_h     = scalar in [0,1], model-period depreciation rate of
                  Home final goods capital
    alpha_f     = scalar in (0, 1), share parameter in CES production
                  function of Foreign intermediate goods producer
    phi_f       = scalar >= 1, elasticity of substitution in CES
                  production function of Foreign intermediate goods
                  producer
    Z_f         = scalar > 0, total factor productivity parameter in
                  Foreign final goods producer production function
    gamma_f     = scalar in (0, 1), capital share of income in Foreign
                  Cobb-Douglas production function
    delta_f     = scalar in [0,1], model-period depreciation rate of
                  Foreign final goods capital
    tol_outer   = scalar > 0, tolerance level for steady-state outer-
                  loop convergence
    tol_inner   = scalar > 0, tolerance level for inner-loop root finder
    xi          = scalar in (0, 1], outer loop updating parameter
    maxiter     = integer >= 1, maximum number of iterations for outer
                  loop fixed point algorithm
    EulDiff     = boolean, =True if use simple differences in Euler
                  errors. Otherwise, use percent deviation form
    results_ol  = length 5 tuple, results from outer_loop() function
    rh_ss       = scalar > 0, steady-state return on Home savings
    rf_ss       = scalar > 0, steady-state return on Foreign savings
    q_ss        = scalar > 0, real exchange rate # Foreign consumption
                  goods per 1 Domestic consumption good
    ol_dist     = scalar > 0, outer loop distance measure
    ol_success  = boolean, =True if outer loop converged
    r_ss        = scalar > -delta_h, steady-state Home composite
                  interest rate
    rstar_ss    = scalar > -delta_f, steady-state Foreign composite
                  interest rate
    KL_rat_h    = scalar > 0, Home final goods capital labor ratio
    KL_rat_f    = scalar > 0, Foreign final goods capital labor ratio
    wh_ss       = scalar > 0, steady-state Home wage
    wf_ss       = scalar > 0, steady-state Foreign wage
    r_h_path    = (S,) vector, steady-state Home composite interest rate
                  over the periods of an agent's life
    w_h_path    = (S,) vector, steady-state Home wage over the periods
                  of an agent's life
    cbh_args    = length 6 tuple, arguments to pass into hh.get_cbvec()
    ch_ss       = (S,) vector, steady-state Home household consumption
                  for each age (c_1, c_2, ...c_S)
    bh_ss       = (S-1,) vector, steady-state Home household savings
                  for each age (b_2, b_3, ...b_S)
    bhss_errors = (S-1,) vector, steady-state Home household Euler
                  equation errors for savings decision
    success_h   = boolean, =True if root finder for inner loop Home
                  household savings decision converged
    r_f_path    = (S,) vector, steady-state Foreign composite interest
                  rate over the periods of an agent's life
    w_f_path    = (S,) vector, steady-state Foreign wage over the
                  periods of an agent's life
    cbf_args    = length 6 tuple, arguments to pass into hh.get_cbvec()
    cf_ss       = (S,) vector, steady-state Foreign household
                  consumption for each age (c_1, c_2, ...c_S)
    bf_ss       = (S-1,) vector, steady-state Foreign household savings
                  for each age (b_2, b_3, ...b_S)
    bfss_errors = (S-1,) vector, steady-state Foreign household Euler
                  equation errors for savings decision
    success_f   = boolean, =True if root finder for inner loop Foreign
                  household savings decision converged
    L_h_ss      = scalar > 0, steady-state Home aggregate labor
    L_f_ss      = scalar > 0, steady-state Foreign aggregate labor
    K_h_ss      = scalar > 0, steady-state Home final goods capital
                  stock
    K_f_ss      = scalar > 0, steady-state Foreign final goods capital
                  stock
    Yh_ss       = scalar > 0, steady-state Home aggregate final goods
                  output (GDP)
    Yf_ss       = scalar > 0, steady-state Foreign aggregate final goods
                  output (GDP)
    K_hh_ss     = scalar > 0, steady-state total Home country savings
                  allocated to Home country intermed't goods production
    K_ff_ss     = scalar > 0, steady-state total Foreign country savings
                  allocated to Foreign country int'd't goods production
    K_fh_ss     = scalar > 0, steady-state total Home country savings
                  allocated to Foreign country int'd't goods production
    K_hf_ss     = scalar > 0, steady-state total Foreign country savings
                  allocated to Home country intermed't goods production
    Ch_ss       = scalar > 0, steady-state Home aggregate consumption
    Cf_ss       = scalar > 0, steady-state Foreign aggregate consumption
    Ih_ss       = scalar, steady-state Home aggregate investment
    If_ss       = scalar, steady-state Foreign aggregate investment
    NXh_ss      = scalar, steady-state Home net exports
    NXf_ss      = scalar, steady-state Foreign net exports
    RC_h_err_ss = scalar, steady-state Home goods market clearing
                  (resource constraint) error
    RC_f_err_ss = scalar, steady-state Foreign goods market clearing
                  (resource constraint) error
    ss_time     = scalar > 0, time elapsed during SS computation
                  (in seconds)
    ss_output   = length 30 dict, steady-state equilibrium objects
                  {bh_ss, ch_ss, bhss_errors, bf_ss, cf_ss, bfss_errors,
                  wh_ss, rh_ss, r_ss, q_ss, wf_ss, rf_ss, rstar_ss,
                  L_h_ss, K_h_ss, K_hh_ss, K_fh_ss, Yh_ss, Ih_ss,
                  NXh_ss, L_f_ss, K_f_ss, K_ff_ss, K_hf_ss, Yf_ss,
                  If_ss, NXf_ss, RC_h_err_ss, RC_f_err_ss, ss_time}

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
    Ch_ss = aggr.get_C(ch_ss)
    Cf_ss = aggr.get_C(cf_ss)
    # Solve for Home and Foreign aggregate investment
    Ih_ss = aggr.get_I(K_h_ss, K_h_ss, delta_h)
    If_ss = aggr.get_I(K_f_ss, K_f_ss, delta_f)
    # Solve for Home and Foreign net exports
    NXh_ss = aggr.get_NX(K_hh_ss, K_fh_ss, K_hh_ss, K_fh_ss, K_h_ss,
                         K_h_ss, r_ss, rh_ss)
    NXf_ss = aggr.get_NX(K_ff_ss, K_hf_ss, K_ff_ss, K_hf_ss, K_f_ss,
                         K_f_ss, rstar_ss, rf_ss)
    # NXh_ss = r_ss * K_h_ss - rh_ss * (K_hh_ss + K_fh_ss)
    # NXf_ss = rstar_ss * K_f_ss - rf_ss * (K_ff_ss + K_hf_ss)
    # Solve for Home and Foreign goods market clearing (resource
    # constraint) errors
    RC_h_err_ss = Yh_ss - Ch_ss - Ih_ss - NXh_ss
    RC_f_err_ss = Yf_ss - Cf_ss - If_ss - NXf_ss

    ss_time = time.clock() - start_time

    ss_output = \
        {'bh_ss': bh_ss, 'ch_ss': ch_ss, 'bhss_errors': bhss_errors,
         'bf_ss': bf_ss, 'cf_ss': cf_ss, 'bfss_errors': bfss_errors,
         'wh_ss': wh_ss, 'rh_ss': rh_ss, 'r_ss': r_ss, 'q_ss': q_ss,
         'wf_ss': wf_ss, 'rf_ss': rf_ss, 'rstar_ss': rstar_ss,
         'L_h_ss': L_h_ss, 'K_h_ss': K_h_ss, 'K_hh_ss': K_hh_ss,
         'K_fh_ss': K_fh_ss, 'Yh_ss': Yh_ss, 'Ih_ss': Ih_ss,
         'NXh_ss': NXh_ss, 'L_f_ss': L_f_ss, 'K_f_ss': K_f_ss,
         'K_ff_ss': K_ff_ss, 'K_hf_ss': K_hf_ss, 'Yf_ss': Yf_ss,
         'If_ss': If_ss, 'NXf_ss': NXf_ss, 'RC_h_err_ss': RC_h_err_ss,
         'RC_f_err_ss': RC_f_err_ss, 'ss_time': ss_time}

    with np.printoptions(precision=4):
        print('bh_ss is: ', bh_ss)
        print('bf_ss is: ', bf_ss)
    print('K_h_ss=', '%10.4f' % K_h_ss, ', rh_ss=', '%10.4f' % rh_ss,
          ', r_ss=', '%10.4f' % r_ss, ', wh_ss=', '%10.4f' % wh_ss)
    print('K_f_ss=', '%10.4f' % K_f_ss, ', rf_ss=', '%10.4f' % rf_ss,
          ', r*_ss=', '%10.4f' % rstar_ss,
          ', wf_ss=', '%10.4f' % wf_ss)
    print('Real exchange rate=', '%10.4f' % q_ss)
    print('Max. abs. savings Euler error is: ', '%10.4e' %
          np.absolute(np.append(bhss_errors, bfss_errors)).max())
    print('Max. abs. resource constraint error is: ', '%10.4e' %
          np.absolute(np.append(RC_h_err_ss, RC_f_err_ss)).max())

    # Print SS computation time
    utils.print_time(ss_time, 'SS')

    if graphs:
        get_ss_graphs(ch_ss, bh_ss, home=True)
        get_ss_graphs(cf_ss, bf_ss, home=False)

    return ss_output
