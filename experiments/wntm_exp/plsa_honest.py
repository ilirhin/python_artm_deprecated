#coding: utf-8
from general_exp import *
from artm_utils.optimizations import em_optimization
from artm_utils.optimizations import get_prob_matrix_by_counters


def regularization_function(phi, theta, n_tw, n_dt):
    n_w = np.sum(n_tw, axis=0)
    phi_matrix = get_prob_matrix_by_counters(n_tw)
    theta_matrix = get_prob_matrix_by_counters(np.transpose(phi_matrix))

    ans = - n_w * theta_matrix.T / 2.
    print np.max(n_tw), np.min(ans)
    return ans, - n_dt  + np.transpose(n_tw + ans)

n_dw_matrix = load_data()

perform_experiment(n_dw_matrix, em_optimization, regularization_function, './plsa_honest')
