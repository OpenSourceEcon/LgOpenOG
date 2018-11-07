'''
------------------------------------------------------------------------
This script tests the functions of the household.py module in the large
open economy two-country OG chapter code. Specifically, it tests:

1) Levels, first derivatives, and second derivatives are equal at
   epsilon in hybrid functions
2) Hybrid functions work for scalars and vectors with values on both
   sides of epsilon and for different parameter values
------------------------------------------------------------------------
'''

# Import packages
import pytest
import numpy as np
import household as hh

'''
------------------------------------------------------------------------
Things to test (still to do):
1) asdf
------------------------------------------------------------------------
'''

bvec_init = np.array([0.02, 0.022, 0.018])
rpath = 0.03 * np.ones(4)
wpath = 1.25 * np.ones(4)
b_cur = 0.0
nvec = np.ones(4)
beta_annual = 0.96
beta = beta_annual ** (80 / 4)
sigma = 2.2
simp_diff = False
cb_args = (b_cur, nvec, beta, sigma, simp_diff)
def test_get_cbvec(bvec_init, rpath, wpath, cb_args):
    bvec, cvec, success = hh.get_cbvec(bvec_init, rpath, wpath, cb_args)
    assert success
    assert (cvec > 0).sum() == cvec.shape[0]


# @pytest.mark.parametrize('bvec_init',
#                          [4.0, 0.1, -0.02, np.array([0.01, 0.02, 0.03]),
#                           np.array([0.01, 0.02, 0.021, 0.022, 0.023,
#                                     0.024, 0.025, 0.026, 0.027, 0.028,
#                                     0.029, 0.030, 0.031, 0.031, 0.031,
#                                     0.032, 0.032, 0.032, 0.032, 0.032,
#                                     0.033, 0.033, 0.033, 0.033, 0.033,
#                                     0.033, 0.033, 0.034, 0.034, 0.034,
#                                     0.034, 0.034, 0.034, 0.034, 0.034,
#                                     0.033, 0.033, 0.033, 0.033, 0.033,
#                                     0.033, 0.032, 0.032, 0.032, 0.032,
#                                     0.031, 0.031, 0.031, 0.031, 0.030,
#                                     0.030, 0.030, 0.029, 0.029, 0.029,
#                                     0.028, 0.028, 0.027, 0.027, 0.026,
#                                     0.026, 0.025, 0.025, 0.024, 0.023,
#                                     0.022, 0.021, 0.020, 0.019, 0.018])])




# def test_get_cbvec(bvec_init, rpath, wpath, cb_args)

# @pytest.mark.parametrize('c_vec',
#                          [np.array([-1.0, -0.5, 2.0, -0.7, -1.3, 0.2,
#                                     5.0, -0.5]),
#                           np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
#                                     5.0, 5.0]),
#                           np.array([9.8, 100.7, 32.3, 59.0, 1.0, 0.1,
#                                     0.00001, 5.0])])
# @pytest.mark.parametrize('cmin_vec',
#                          [np.array([0.0, 0.4, 0.2, 0.5, 0.1, 0.0,
#                                     0.5, 1.0]),
#                           np.array([0.01, -0.1, 0.01, -0.1, 0.01, -0.1,
#                                     0.01, -0.1]),
#                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                                     0.00002, 0.0])])
# @pytest.mark.parametrize('share_vec',
#                          [np.array([0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
#                                     0.2, 0.29]),
#                           np.array([0.125, 0.125, 0.125, 0.125, 0.125,
#                                     0.125, 0.125, 0.125]),
#                           np.array([0.29, 0.2, 0.11, 0.1, 0.09, 0.08,
#                                     0.07, 0.06])])
# @pytest.mark.parametrize('eta5', [1.5, 4.2, 5.0])
# def test_CEScomp(c_vec, cmin_vec, share_vec, eta5):
#     '''
#     --------------------------------------------------------------------
#     This test that various consumption vectors, cmin vectors, share
#     vectors and eta values give finite composite consumption values
#     --------------------------------------------------------------------
#     '''
#     comp_C = hh.CEScomp_cons(c_vec, cmin_vec, share_vec, eta5)
#     assert (comp_C > -np.infty and comp_C < np.infty)


# @pytest.mark.parametrize('epsilon_2', [0.0001, 0.01, 0.1])
# @pytest.mark.parametrize('gamma1', [1.0, 2.1, 3.0])
# def test_CRRA_levder(epsilon_2, gamma1):
#     '''
#     --------------------------------------------------------------------
#     This test verifies the level and derivative conditions for the u(C)
#     CRRA utility function for different values of epsilon_2 and gamma
#     --------------------------------------------------------------------
#     '''
#     # Test that f_1(epsilon_2) = f_2(epsilon_2)
#     assert np.isclose(hh.f_1(epsilon_2, gamma1),
#                       hh.f_2(epsilon_2, gamma1, epsilon_2))
#     # Test that f_1'(epsilon_2) = f_2'(epsilon_2)
#     f_1_der_eps = epsilon_2 ** (-gamma1)
#     d2 = -gamma1 * (epsilon_2 ** (-1 - gamma1))
#     d1 = (epsilon_2 ** (-gamma1)) * (1 + gamma1)
#     f_2_der_eps = d1 + d2 * epsilon_2
#     assert np.isclose(f_1_der_eps, f_2_der_eps)
#     # Test that f_1''(epsilon_2) = f_2''(epsilon_2)
#     f_1_der2_eps = -gamma1 * (epsilon_2 ** (-1 - gamma1))
#     f_2_der2_eps = d2
#     assert np.isclose(f_1_der2_eps, f_2_der2_eps)


# @pytest.mark.parametrize('gamma2', [1.0, 2.1, 3.0])
# @pytest.mark.parametrize('u_Cvals',
#                          [-0.1, 2.0, np.array([-0.1, 0.0, 0.0001, 0.2,
#                                                5.2])])
# def test_u_Cvals(u_Cvals, gamma2):
#     '''
#     --------------------------------------------------------------------
#     This test verifies the output of the CRRA_util u(C) function for
#     varying values of C in both scalar and vector form and varying
#     values of gamma
#     --------------------------------------------------------------------
#     '''
#     epsilon_2 = 0.01
#     d2 = -gamma2 * (epsilon_2 ** (-1 - gamma2))
#     d1 = (epsilon_2 ** (-gamma2)) * (1 + gamma2)
#     if gamma2 == 1.0:
#         d0 = np.log(epsilon_2) - (3 / 2)
#     elif gamma2 > 1.0:
#         d0 = \
#             ((gamma2 * (epsilon_2 ** (1 - gamma2)) * (1 + gamma2) - 2) /
#              (2 * (1 - gamma2)))
#     u_vals = hh.CRRA_util(u_Cvals, gamma2)
#     if np.isscalar(u_vals):
#         if u_Cvals >= epsilon_2:
#             if gamma2 == 1.0:
#                 utility = np.log(u_Cvals)
#             elif gamma2 > 1.0:
#                 utility = ((u_Cvals ** (1 - gamma2)) - 1) / (1 - gamma2)
#             assert np.isclose(u_vals, utility)
#         elif u_Cvals < epsilon_2:
#             utility = d0 + d1 * u_Cvals + 0.5 * d2 * u_Cvals ** 2
#             assert np.isclose(u_vals, utility)
#     else:
#         u_Cvals_neg = u_Cvals < epsilon_2
#         uvals_neg = u_vals[u_Cvals_neg]
#         uvals_pos = u_vals[~u_Cvals_neg]
#         u_negfunvals = (d0 + d1 * u_Cvals[u_Cvals_neg] +
#                         0.5 * d2 * u_Cvals[u_Cvals_neg] ** 2)
#         if gamma2 == 1.0:
#             u_posfunvals = np.log(u_Cvals[~u_Cvals_neg])
#         elif gamma2 > 1.0:
#             u_posfunvals = (((u_Cvals[~u_Cvals_neg] ** (1 - gamma2)) - 1) /
#                             (1 - gamma2))
#         assert (np.isclose(uvals_neg, u_negfunvals).sum() ==
#                 uvals_neg.size)
#         assert (np.isclose(uvals_pos, u_posfunvals).sum() ==
#                 uvals_pos.size)


# @pytest.mark.parametrize('epsilon_0', [0.0001, 0.001, 0.1])
# @pytest.mark.parametrize('eta1', [1.5, 4.2, 5.0])
# def test_gc_levder(epsilon_0, eta1):
#     '''
#     --------------------------------------------------------------------
#     This test verifies the level and derivative conditions for the g(c)
#     function for different values of epsilon_0 and eta
#     --------------------------------------------------------------------
#     '''
#     # Test that g_1(epsilon_0) = g_2(epsilon_0)
#     assert np.isclose(hh.g_1(epsilon_0, eta1),
#                       hh.g_2(epsilon_0, eta1, epsilon_0))
#     # Test that g_1'(epsilon_0) = g_2'(epsilon_0)
#     g_1_der_eps = ((eta1 - 1) / eta1) * epsilon_0 ** (-1 / eta1)
#     a1 = (((eta1 ** 2) - 1) / (eta1 ** 2)) * (epsilon_0 ** (-1 / eta1))
#     a2 = -((eta1 - 1) / (eta1 ** 2)) * (epsilon_0 ** (-(eta1 + 1) /
#                                                       eta1))
#     g_2_der_eps = a1 + a2 * epsilon_0
#     assert np.isclose(g_1_der_eps, g_2_der_eps)
#     # Test that g_1''(epsilon_0) = g_2''(epsilon_0)
#     g_1_der2_eps = (-((eta1 - 1) / (eta1 ** 2)) *
#                     epsilon_0 ** (-(eta1 + 1) / eta1))
#     g_2_der2_eps = a2
#     assert np.isclose(g_1_der2_eps, g_2_der2_eps)


# @pytest.mark.parametrize('eta2', [1.5, 4.2, 5.0])
# @pytest.mark.parametrize('gc_cvals',
#                          [-0.1, 2.0, np.array([-0.1, 0.0, 0.0001, 0.2,
#                                                5.2])])
# def test_gc_cvals(gc_cvals, eta2):
#     '''
#     --------------------------------------------------------------------
#     This test verifies the output of the g(c) function for varying
#     values of c_min_vec in both scalar and vector form and varying
#     values of eta
#     --------------------------------------------------------------------
#     '''
#     epsilon_0 = 0.001
#     a0 = (((1 + eta2) / (2 * (eta2 ** 2))) *
#           (epsilon_0 ** ((eta2 - 1) / eta2)))
#     a1 = (((eta2 ** 2) - 1) / (eta2 ** 2)) * (epsilon_0 ** (-1 / eta2))
#     a2 = -((eta2 - 1) / (eta2 ** 2)) * (epsilon_0 ** (-(eta2 + 1) /
#                                                       eta2))
#     g_vals = hh.g_c(gc_cvals, eta2)
#     if np.isscalar(g_vals):
#         if gc_cvals >= epsilon_0:
#             assert np.isclose(g_vals, gc_cvals ** ((eta2 - 1) / eta2))
#         elif gc_cvals < epsilon_0:
#             assert np.isclose(g_vals, (a0 + a1 * gc_cvals +
#                                        0.5 * a2 * gc_cvals ** 2))
#     else:
#         gc_cvals_neg = gc_cvals < epsilon_0
#         gvals_neg = g_vals[gc_cvals_neg]
#         gvals_pos = g_vals[~gc_cvals_neg]
#         assert (np.isclose(gvals_neg,
#                            (a0 + a1 * gc_cvals[gc_cvals_neg] +
#                             0.5 * a2 *
#                             gc_cvals[gc_cvals_neg] ** 2)).sum() ==
#                 gvals_neg.size)
#         assert (np.isclose(gvals_pos, (gc_cvals[~gc_cvals_neg] **
#                                        ((eta2 - 1) / eta2))).sum() ==
#                 gvals_pos.size)


# @pytest.mark.parametrize('epsilon_1', [0.0001, 0.001, 0.1])
# @pytest.mark.parametrize('eta3', [1.5, 4.2, 5.0])
# def test_hs_levder(epsilon_1, eta3):
#     '''
#     --------------------------------------------------------------------
#     This test verifies the level and derivative conditions for the
#     h(in_sum) function for different values of epsilon_1 and eta
#     --------------------------------------------------------------------
#     '''
#     # Test that h_1(epsilon_1) = h_2(epsilon_1)
#     assert np.isclose(hh.h_1(epsilon_1, eta3),
#                       hh.h_2(epsilon_1, eta3, epsilon_1))
#     # Test that h_1'(epsilon_1) = h_2'(epsilon_1)
#     h_1_der_eps = (eta3 / (eta3 - 1)) * epsilon_1 ** (1 / (eta3 - 1))
#     slope_pct = 0.8
#     h1eps_slope = (eta3 / (eta3 - 1)) * (epsilon_1 ** (1 / (eta3 - 1)))
#     b2 = 1 / ((1 - slope_pct) * epsilon_1 * (eta3 - 1))
#     b1 = (((eta3 / (eta3 - 1)) * (1 - slope_pct) *
#            (epsilon_1 ** (1 / (eta3 - 1)))) /
#           (b2 * np.exp(b2 * epsilon_1)))
#     h_2_der_eps = (slope_pct * h1eps_slope +
#                    b2 * b1 * np.exp(b2 * epsilon_1))
#     assert np.isclose(h_1_der_eps, h_2_der_eps)
#     # Test that h_1''(epsilon_1) = h_2''(epsilon_1)
#     h_1_der2_eps = ((eta3 / ((eta3 - 1) ** 2)) *
#                     (epsilon_1 ** ((2 - eta3) / (eta3 - 1))))
#     h_2_der2_eps = (b2 ** 2) * b1 * np.exp(b2 * epsilon_1)
#     assert np.isclose(h_1_der2_eps, h_2_der2_eps)


# @pytest.mark.parametrize('eta4', [1.5, 4.2, 5.0])
# @pytest.mark.parametrize('hs_cvals',
#                          [-0.1, 2.0, np.array([-0.1, 0.0, 0.0001, 0.2,
#                                                5.2])])
# def test_hc_svals(hs_cvals, eta4):
#     '''
#     --------------------------------------------------------------------
#     This test verifies the output of the h(in_sum) function for varying
#     values of in_sum in both scalar and vector form and varying
#     values of eta
#     --------------------------------------------------------------------
#     '''
#     epsilon_1 = 0.001
#     slope_pct = 0.8
#     h1eps_slope = (eta4 / (eta4 - 1)) * (epsilon_1 ** (1 / (eta4 - 1)))
#     b2 = 1 / ((1 - slope_pct) * epsilon_1 * (eta4 - 1))
#     b1 = (((eta4 / (eta4 - 1)) * (1 - slope_pct) *
#            (epsilon_1 ** (1 / (eta4 - 1)))) /
#           (b2 * np.exp(b2 * epsilon_1)))
#     b0 = ((epsilon_1 ** (eta4 / (eta4 - 1))) *
#           (1 - slope_pct * (eta4 / (eta4 - 1))) -
#           b1 * np.exp(b2 * epsilon_1))
#     h_vals = hh.h_C(hs_cvals, eta4)
#     if np.isscalar(h_vals):
#         if hs_cvals >= epsilon_1:
#             assert np.isclose(h_vals, hs_cvals ** (eta4 / (eta4 - 1)))
#         elif hs_cvals < epsilon_1:
#             hs_funval = (slope_pct * h1eps_slope * hs_cvals + b0 + b1 *
#                          np.exp(b2 * hs_cvals))
#             assert np.isclose(h_vals, hs_funval)
#     else:
#         hs_cvals_neg = hs_cvals < epsilon_1
#         hvals_neg = h_vals[hs_cvals_neg]
#         hvals_pos = h_vals[~hs_cvals_neg]
#         hs_funvals = (slope_pct * h1eps_slope * hs_cvals[hs_cvals_neg] +
#                       b0 + b1 * np.exp(b2 * hs_cvals[hs_cvals_neg]))
#         assert (np.isclose(hvals_neg, hs_funvals).sum() ==
#                 hvals_neg.size)
#         assert (np.isclose(hvals_pos, (hs_cvals[~hs_cvals_neg] **
#                                        (eta4 / (eta4 - 1)))).sum() ==
#                 hvals_pos.size)


# @pytest.mark.parametrize('cmin_vec2',
#                          [np.array([0.0, 0.4, 0.2, 0.5, 0.1, 0.0,
#                                     0.5, 1.0]),
#                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                                     0.0, 0.0])])
# @pytest.mark.parametrize('share_vec2',
#                          [np.array([0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
#                                     0.2, 0.29]),
#                           np.array([0.125, 0.125, 0.125, 0.125, 0.125,
#                                     0.125, 0.125, 0.125]),
#                           np.array([0.29, 0.2, 0.11, 0.1, 0.09, 0.08,
#                                     0.07, 0.06])])
# @pytest.mark.parametrize('gamma3', [1.0, 2.1, 3.0])
# @pytest.mark.parametrize('eta6', [1.5, 4.2, 5.0])
# @pytest.mark.parametrize('w1', [0.05, 1.0, 10.0])
# @pytest.mark.parametrize('tau_vec',
#                          [np.array([0.01, -0.02, 0.08, 0.09, 0.1, 0.0,
#                                     0.2, -0.02]),
#                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                                     0.0, 0.0]),
#                           np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
#                                     0.4, 0.4])])
# def test_util_max(cmin_vec2, share_vec2, gamma3, eta6, w1, tau_vec):
#     '''
#     --------------------------------------------------------------------
#     This test verifies that the utility maximization function util_max()
#     works for a large grid of potential input values
#     --------------------------------------------------------------------
#     '''
#     pref_args = (cmin_vec2, share_vec2, gamma3)
#     type_args = (eta6, w1)
#     policy_args = tau_vec

#     c_opt, u_opt, tax_opt, below_min, success = \
#         hh.util_max(pref_args, type_args, policy_args)
#     assert success
#     if ((1 + tau_vec) * cmin_vec2).sum() >= w1:
#         assert below_min
#     assert (c_opt <= 0).sum() == 0
#     assert (c_opt * tau_vec).sum() == tax_opt
#     assert hh.u_c(c_opt, cmin_vec2, share_vec2, gamma3, eta6, w1,
#                   tau_vec) == u_opt


# @pytest.mark.parametrize('cmin_vec3',
#                          [np.array([0.0, 0.4, 0.2, 0.5, 0.1, 0.0,
#                                     0.5, 1.0]),
#                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                                     0.0, 0.0])])
# @pytest.mark.parametrize('share_vec3',
#                          [np.array([0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
#                                     0.2, 0.29]),
#                           np.array([0.125, 0.125, 0.125, 0.125, 0.125,
#                                     0.125, 0.125, 0.125]),
#                           np.array([0.29, 0.2, 0.11, 0.1, 0.09, 0.08,
#                                     0.07, 0.06])])
# @pytest.mark.parametrize('w2', [0.05, 1.0, 10.0])
# @pytest.mark.parametrize('tau_vec2',
#                          [np.array([0.01, -0.02, 0.08, 0.09, 0.1, 0.0,
#                                     0.2, -0.02]),
#                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                                     0.0, 0.0]),
#                           np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
#                                     0.4, 0.4])])
# def test_init_guess(cmin_vec3, share_vec3, w2, tau_vec2):
#     '''
#     --------------------------------------------------------------------
#     This test verifies that the initial guess function init_guess()
#     works for a large grid of potential input values
#     --------------------------------------------------------------------
#     '''
#     cvec_init, below_min = hh.init_guess(cmin_vec3, share_vec3, w2,
#                                          tau_vec2)
#     if (cvec_init <= 0).sum() == 0:
#         print('cvec_init=', cvec_init)
#         print('cmin_vec=', cmin_vec3)
#         print('w=', w2)
#         print('tau_vec=', tau_vec2)
#     assert (cvec_init <= 0).sum() == 0
#     if ((1 + tau_vec2) * cmin_vec3).sum() >= w2:
#         assert below_min
#     else:
#         assert not below_min
