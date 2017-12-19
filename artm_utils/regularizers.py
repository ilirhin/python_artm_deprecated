# coding: utf-8

import numpy as np
from optimizations import get_prob_matrix_by_counters


def trivial_regularization(phi, theta, n_tw, n_dt):
    return np.zeros_like(n_tw), np.zeros_like(n_dt)


def create_reg_decorr(tau, phi_alpha=0., theta_alpha=0., use_old_phi=False):
    def fun(phi, theta, n_tw, n_dt):
        T, _ = phi.shape
        if not use_old_phi:
            phi_matrix = get_prob_matrix_by_counters(n_tw)
        else:
            phi_matrix = phi
        aggr_phi = np.sum(phi_matrix, axis=0)
        return - 1. / (T - 1) / T * tau * phi_matrix * (aggr_phi - phi_matrix), theta_alpha

    return fun


def create_reg_lda(phi_alpha, theta_alpha):
    def fun(phi, theta, n_tw, n_dt):
        return np.zeros_like(n_tw) + phi_alpha, np.zeros_like(n_dt) + theta_alpha

    return fun
