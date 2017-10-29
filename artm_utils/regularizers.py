# coding: utf-8

import numpy as np


def trivial_regularization(phi, theta, n_tw, n_dt):
    return np.zeros_like(n_tw), np.zeros_like(n_dt)


def create_reg_decorr(tau, theta_alpha=0.):
    def fun(phi, theta, n_tw, n_dt):
        phi_matrix = n_tw / np.sum(n_tw, axis=1)[:, np.newaxis]
        aggr_phi = np.sum(phi_matrix, axis=1)
        return - tau * np.transpose(phi_matrix * (aggr_phi[:, np.newaxis] - phi_matrix)), theta_alpha

    return fun


def create_reg_lda(phi_alpha, theta_alpha):
    def fun(phi, theta, n_tw, n_dt):
        return np.zeros_like(n_tw) + phi_alpha, np.zeros_like(n_dt) + theta_alpha

    return fun
