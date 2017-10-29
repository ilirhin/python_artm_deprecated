#coding: utf-8
from general_exp import *
from artm_utils.optimizations import em_optimization
from artm_utils.optimizations import get_prob_matrix_by_counters


def regularization_function(phi, theta, n_tw, n_dt):
    theta_matrix = get_prob_matrix_by_counters(np.transpose(n_tw))
    n_w = np.sum(n_tw, axis=0)

    ans = - n_w * theta_matrix / 2.
    return ans, - n_dt  + np.transpose(n_tw + ans)

n_dw_matrix = load_data()

perform_experiment(n_dw_matrix, em_optimization, regularization_function, './plsa_honest')
