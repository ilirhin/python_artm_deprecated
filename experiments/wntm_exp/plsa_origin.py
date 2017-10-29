#coding: utf-8
from general_exp import *
from artm_utils.optimizations import em_optimization


def regularization_function(phi, theta, n_tw, n_dt):
    return 0., 0.

n_dw_matrix = load_data()

perform_experiment(n_dw_matrix, em_optimization, regularization_function, './plsa_origin')