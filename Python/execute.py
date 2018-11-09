'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the model with 2 countries, S-period lived agents
in each, and exogenous labor from the large open chapter of the OG
textbook.

This Python script imports the following module(s):
    SS.py

    TPI.py
    aggregates.py
    utilities.py

This Python script calls the following function(s):
    ss.get_SS()
    utils.compare_args()

    aggr.get_K()
    tpi.get_TPI()

Files created by this script:
    OUTPUT/SS/ss_vars.pkl
    OUTPUT/SS/ss_args.pkl
    OUTPUT/TPI/tpi_vars.pkl
    OUTPUT/TPI/tpi_args.pkl
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import pickle
import os
import SS as ss
# import household as hh
# import TPI as tpi
# import aggregates as aggr
import utilities as utils

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S              = integer in [3,80], number of periods an individual
                 lives
beta_annual    = scalar in (0,1), discount factor for one year
beta           = scalar in (0,1), discount factor for each model period
sigma          = scalar > 0, coefficient of relative risk aversion
nhcutper       = int >= 2, age at which Home labor supply is exogenously
                 reduced
nhvec          = (S,) vector, exogenous Home labor supply n_{h,s,t}
nfcutper       = int >= 2, age at which Foreign labor supply is
                 exogenously reduced
nfvec          = [S,] vector, exogenous Foreign labor supply n_{f,s,t}
alpha_h        = scalar in (0, 1), share parameter in CES production
                 function of Home intermediate goods producer
phi_h          = scalar >= 1, elasticity of substitution in CES
                 production function of Home intermediate goods producer
Z_h            = scalar > 0, total factor productivity parameter in
                 Home final goods producer production function
gamma_h        = scalar in (0, 1), capital share of income in Home Cobb-
                 Douglas production function
delta_h_annual = scalar in [0,1], one-year depreciation rate of Home
                 final goods capital
delta_h        = scalar in [0,1], model-period depreciation rate of Home
                 final goods capital
alpha_f        = scalar in (0, 1), share parameter in CES production
                 function of Foreign intermediate goods producer
phi_f          = scalar >= 1, elasticity of substitution in CES prod
                 function of Foreign intermediate goods producer
Z_f            = scalar > 0, total factor productivity parameter in
                 Foreign final goods producer production function
gamma_f        = scalar in (0, 1), capital share of income in Foreign
                 Cobb-Douglas production function
delta_f_annual = scalar in [0,1], one-year depreciation rate of Foreign
                 final goods capital
delta_f        = scalar in [0,1], model-period depreciation rate of
                 Foreign final goods capital
SS_solve       = boolean, =True if want to solve for steady-state
                 solution, otherwise retrieve solutions from pickle
SS_maxiter     = integer >= 1, maximum number of iterations for steady-
                 state outer loop fixed point algorithm
SS_tol_outer   = scalar > 0, tolerance level for steady-state outer-loop
                 convergence
SS_tol_inner   = scalar > 0, tolerance level for inner-loop root finder
xi_SS          = scalar in (0, 1], steady-state outer loop updating
                 parameter
SS_graphs     = boolean, =True if want graphs of steady-state objects
SS_EulDiff    = boolean, =True if use simple differences in Euler
                errors. Otherwise, use percent deviation form
T             = integer > S, number of time periods until steady state
TPI_solve     = boolean, =True if want to solve TPI after solving SS
TPI_tol       = scalar > 0, tolerance level for fsolve's in TPI
maxiter_TPI   = integer >= 1, Maximum number of iterations for TPI
mindist_TPI   = scalar > 0, Convergence criterion for TPI
xi_TPI        = scalar in (0,1], TPI path updating parameter
TPI_graphs    = Boolean, =True if want graphs of TPI objects
TPI_EulDiff   = Boolean, =True if want difference version of Euler
                errors beta*(1+r)*u'(c2) - u'(c1), =False if want ratio
                version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
------------------------------------------------------------------------
'''
# Both countries household parameters
S = int(80)
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 2.2

# Home country household parameters
nhcutper = round((9 / 16) * S)
nhvec = np.ones(S)
nhvec[nhcutper:] = 0.2

# Foreign country household parameters
nfcutper = round((9 / 16) * S)
nfvec = np.ones(S)
nfvec[nfcutper:] = 0.2

# Home country intermediate goods producers parameters
alpha_h = 0.6
phi_h = 2.5

# Home country final goods producers parameters
Z_h = 1.0
gamma_h = 0.3
delta_h_annual = 0.05
delta_h = 1 - ((1 - delta_h_annual) ** (80 / S))

# Foreign country intermediate goods producers parameters
alpha_f = 0.55
phi_f = 2.1

# Foreign country final goods producers parameters
Z_f = 1.0
gamma_f = 0.32
delta_f_annual = 0.04
delta_f = 1 - ((1 - delta_f_annual) ** (80 / S))

# SS parameters
SS_solve = True
SS_maxiter = 200
SS_tol_outer = 1e-13
SS_tol_inner = 1e-13
xi_SS = 0.2
SS_graphs = True
SS_EulDiff = True
# TPI parameters
T = int(round(3.0 * S))
TPI_solve = True
TPI_tol = 1e-13
maxiter_TPI = 200
mindist_TPI = 1e-13
xi_TPI = 0.20
TPI_graphs = True
TPI_EulDiff = True

'''
------------------------------------------------------------------------
Solve for the steady-state solution
------------------------------------------------------------------------
cur_path       = string, current file path of this script
ss_output_fldr = string, cur_path extension of SS output folder path
ss_output_dir  = string, full path name of SS output folder
ss_outputfile  = string, path name of file for SS output objects
ss_paramsfile  = string, path name of file for SS parameter objects
rh_ss_guess    = scalar > 0, initial guess for rh_ss
rf_ss_guess    = scalar > 0, initial guess for rf_ss
q_ss_guess     = scalar > 0, initial guess for q_ss
ss_args        = length 19 tuple, arguments to be passed in to
                 ss.get_SS()
ss_output      = length 30 dict, steady-state equilibrium objects
                 {bh_ss, ch_ss, bhss_errors, bf_ss, cf_ss, bfss_errors,
                 wh_ss, rh_ss, r_ss, q_ss, wf_ss, rf_ss, rstar_ss,
                 L_h_ss, K_h_ss, K_hh_ss, K_fh_ss, Yh_ss, Ih_ss, NXh_ss,
                 L_f_ss, K_f_ss, K_ff_ss, K_hf_ss, Yf_ss, If_ss, NXf_ss,
                 RC_h_err_ss, RC_f_err_ss, ss_time}
ss_vars_exst   = boolean, =True if ss_vars.pkl exists
ss_args_exst   = boolean, =True if ss_args.pkl exists
cur_ss_args    = length 19 tuple, current args to be passed in to
                 ss.get_SS()
args_same      = boolean, =True if ss_args == cur_ss_args
------------------------------------------------------------------------
'''
# Create OUTPUT/SS directory if does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
ss_output_fldr = 'OUTPUT/SS'
ss_output_dir = os.path.join(cur_path, ss_output_fldr)
if not os.access(ss_output_dir, os.F_OK):
    os.makedirs(ss_output_dir)
ss_outputfile = os.path.join(ss_output_dir, 'ss_vars.pkl')
ss_paramsfile = os.path.join(ss_output_dir, 'ss_args.pkl')

# Compute steady-state solution
if SS_solve:
    print('BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION')
    # Make initial guess for rh_ss, rf_ss, and q_ss
    rh_ss_guess = 0.04
    rf_ss_guess = 0.04
    q_ss_guess = 1.0
    ss_args = (nhvec, nfvec, beta, sigma, alpha_h, phi_h, Z_h, gamma_h,
               delta_h, alpha_f, phi_f, Z_f, gamma_f, delta_f,
               SS_tol_outer, SS_tol_inner, xi_SS, SS_maxiter,
               SS_EulDiff)
    ss_output = ss.get_SS(rh_ss_guess, rf_ss_guess, q_ss_guess, ss_args,
                          SS_graphs)

    # Save ss_output as pickle
    pickle.dump(ss_output, open(ss_outputfile, 'wb'))
    pickle.dump(ss_args, open(ss_paramsfile, 'wb'))

# Don't compute steady-state, get it from pickle
else:
    # Make sure that the SS output files exist
    ss_vars_exst = os.path.exists(ss_outputfile)
    ss_args_exst = os.path.exists(ss_paramsfile)
    if (not ss_vars_exst) or (not ss_args_exst):
        # If the files don't exist, stop the program and run the steady-
        # state solution first
        err_msg = ('ERROR: The SS output files do not exist and ' +
                   'SS_solve=False. Must set SS_solve=True and ' +
                   'compute steady-state solution.')
        raise ValueError(err_msg)
    else:
        # If the files do exist, make sure that none of the parameters
        # changed from the parameters used in the solution for the saved
        # steady-state pickle
        ss_args = pickle.load(open(ss_paramsfile, 'rb'))

        cur_ss_args = \
            (nhvec, nfvec, beta, sigma, alpha_h, phi_h, Z_h, gamma_h,
             delta_h, alpha_f, phi_f, Z_f, gamma_f, delta_f,
             SS_tol_outer, SS_tol_inner, xi_SS, SS_maxiter, SS_EulDiff)

        args_same = utils.compare_args(ss_args, cur_ss_args)
        if args_same:
            # If none of the parameters changed, use saved pickle
            print('RETRIEVE STEADY-STATE SOLUTIONS FROM FILE')
            ss_output = pickle.load(open(ss_outputfile, 'rb'))
        else:
            # If any of the parameters changed, end the program and
            # compute the steady-state solution
            err_msg = ('ERROR: Current ss_args are not equal to the ' +
                       'ss_args that produced ss_output. Must solve ' +
                       'for SS before solving transition path. Set ' +
                       'SS_solve=True.')
            raise ValueError(err_msg)

# '''
# ------------------------------------------------------------------------
# Solve for the transition path equilibrium by time path iteration (TPI)
# ------------------------------------------------------------------------
# tpi_output_fldr = string, cur_path extension of TPI output folder path
# tpi_output_dir  = string, full path name of TPI output folder
# tpi_outputfile  = string, path name of file for TPI output objects
# tpi_paramsfile  = string, path name of file for TPI parameter objects
# K_ss            = scalar > 0, steady-state aggregate capital stock
# L_ss            = scalar > 0, steady-state aggregate labor
# C_ss            = scalar > 0, steady-state aggregate consumption
# b_ss            = (S-1,) vector, steady-state savings distribution
# init_wgts       = (S-1,) vector, weights representing the factor by which
#                   the initial wealth distribution differs from the
#                   steady-state wealth distribution
# bvec1           = (S-1,) vector, initial period savings distribution
# K1              = scalar, initial period aggregate capital stock
# K1_cstr         = boolean, =True if K1 <= 0
# tpi_params      = length 16 tuple, args to pass into tpi.get_TPI()
# tpi_output      = length 14 dictionary, {cpath, npath, bpath, wpath,
#                   rpath, Kpath, Lpath, Ypath, Cpath, bSp1_err_path,
#                   b_err_path, n_err_path, RCerrPath, tpi_time}
# tpi_args        = length 17 tuple, args that were passed in to get_TPI()
#                   including bvec1
# ------------------------------------------------------------------------
# '''
# if TPI_solve:
#     print('BEGIN EQUILIBRIUM TRANSITION PATH COMPUTATION')

#     # Create OUTPUT/TPI directory if does not already exist
#     cur_path = os.path.split(os.path.abspath(__file__))[0]
#     tpi_output_fldr = 'OUTPUT/TPI'
#     tpi_output_dir = os.path.join(cur_path, tpi_output_fldr)
#     if not os.access(tpi_output_dir, os.F_OK):
#         os.makedirs(tpi_output_dir)
#     tpi_outputfile = os.path.join(tpi_output_dir, 'tpi_vars.pkl')
#     tpi_paramsfile = os.path.join(tpi_output_dir, 'tpi_args.pkl')

#     K_ss = ss_output['K_ss']
#     C_ss = ss_output['C_ss']
#     b_ss = ss_output['b_ss']

#     # Choose initial period distribution of wealth (bvec1), which
#     # determines initial period aggregate capital stock
#     init_wgts = ((1.5 - 0.87) / (S - 2)) * np.arange(S - 1) + 0.87
#     bvec1 = init_wgts * b_ss

#     # Make sure init. period distribution is feasible in terms of K
#     K1, K1_cstr = aggr.get_K(bvec1)

#     # If initial bvec1 is not feasible end program
#     if K1_cstr:
#         err_msg = ('ERROR: Initial savings distribution b1vec caused ' +
#                    'caused an infeasible value for K1 ' +
#                    '(K1 < epsilon). Some elements of bvec1 must ' +
#                    'increase.')
#         raise RuntimeError(err_msg)
#     else:
#         tpi_params = (S, T, beta, sigma, nvec, A, alpha, delta, b_ss,
#                       K_ss, C_ss, maxiter_TPI, mindist_TPI, TPI_tol,
#                       xi_TPI, TPI_EulDiff)
#         tpi_output = tpi.get_TPI(bvec1, tpi_params, TPI_graphs)

#         tpi_args = (S, T, beta, sigma, nvec, A, alpha, delta, K_ss,
#                     C_ss, b_ss, maxiter_TPI, mindist_TPI, TPI_tol,
#                     xi_TPI, TPI_EulDiff, bvec1)

#         # Save tpi_output as pickle
#         pickle.dump(tpi_output, open(tpi_outputfile, 'wb'))
#         pickle.dump(tpi_args, open(tpi_paramsfile, 'wb'))
