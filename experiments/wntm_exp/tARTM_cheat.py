#coding: utf-8
from general_exp import *
from artm_utils.optimizations import artm_thetaless_em_optimization
from artm_utils.regularizers import trivial_regularization


n_dw_matrix = load_data()

perform_experiment(n_dw_matrix, artm_thetaless_em_optimization, trivial_regularization, './tARTM_cheat', params={'use_B_cheat': True})