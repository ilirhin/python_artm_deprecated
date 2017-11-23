#coding: utf-8
import os
from artm_utils.optimizations import *
from artm_utils.calculations import *
import numpy as np
from scipy.sparse import csr_matrix
import time
import pickle


def load_data(data_path='~/data.npy', indices_path='~/indices.npy', indptr_path='~/indptr.npy'):
    data = np.load(os.path.expanduser(data_path))
    indices = np.load(os.path.expanduser(indices_path))
    indptr = np.load(os.path.expanduser(indptr_path))

    n_dw_matrix = csr_matrix((data, indices, indptr))
    n_dw_matrix.eliminate_zeros()

    return n_dw_matrix


last_callback_time = time.time()


def exp_callback(it, phi, theta):
    global last_callback_time
    print it, time.time() - last_callback_time
    last_callback_time = time.time()
    print '\tsparsity', 1. * np.sum(phi == 0) / np.sum(phi >= 0)
    print '\ttheta_sparsity', 1. * np.sum(theta == 0) / np.sum(theta >= 0)


def perform_experiment(n_dw_matrix, optimization_function, regularization_function, res_dir, seeds=[777], T=100, iters=60, params=None):
    if params is None:
        params = {}
    params['return_counters'] = True
    
    D, W = n_dw_matrix.shape
    
    for seed in seeds:
        random_gen = np.random.RandomState(seed)
        phi_matrix = get_prob_matrix_by_counters(random_gen.uniform(size=(T, W)).astype(np.float64))
        theta_matrix = get_prob_matrix_by_counters(np.transpose(phi_matrix))

        regularization_list = np.zeros(iters, dtype=object)
        regularization_list[:] = regularization_function
        
        phi, theta, n_tw, n_dt = optimization_function(
            n_dw_matrix=n_dw_matrix, 
            phi_matrix=phi_matrix,
            theta_matrix=theta_matrix,
            regularization_list=regularization_list,
            iters_count=iters,
            iteration_callback=exp_callback,
            params=params
        )

        _train_perplexity = artm_calc_perplexity_factory(n_dw_matrix) 
        properties = {
            'train_perplexity': _train_perplexity(phi, theta),
            'topic_correlation': artm_calc_topic_correlation(phi),
            'sparsity': 1. * np.sum(phi == 0) / np.sum(phi >= 0),
            'theta_sparsity': 1. * np.sum(theta == 0) / np.sum(theta >= 0),
            'kernel_avg_size': np.mean(artm_get_kernels_sizes(phi)),
            'kernel_avg_jacard': artm_get_avg_pairwise_kernels_jacards(phi),

            'top10_avg_jacard': artm_get_avg_top_words_jacards(phi, 10),
            'top50_avg_jacard': artm_get_avg_top_words_jacards(phi, 50),
            'top100_avg_jacard': artm_get_avg_top_words_jacards(phi, 100),
            'top200_avg_jacard': artm_get_avg_top_words_jacards(phi, 200)
        }

        print 'train_perplexity', properties['train_perplexity']
        print 'topic_correlation', properties['topic_correlation']
        print 'sparsity', properties['sparsity']
        print 'theta_sparsity', properties['theta_sparsity']
        print 'kernel_avg_size', properties['kernel_avg_size']
        print 'kernel_avg_jacard', properties['kernel_avg_jacard']

        print 'top10_avg_jacard', properties['top10_avg_jacard']
        print 'top50_avg_jacard', properties['top50_avg_jacard']
        print 'top100_avg_jacard', properties['top100_avg_jacard']
        print 'top200_avg_jacard', properties['top200_avg_jacard']

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        with open(res_dir + '/exp_seed_{}_topics_{}_iters_{}.pkl'.format(seed, T, iters), 'w') as f:
            pickle.dump({
                'phi': phi, 
                'theta': theta, 
                'n_tw': n_tw, 
                'n_dt': n_dt,
                'properties': properties
            }, f)

