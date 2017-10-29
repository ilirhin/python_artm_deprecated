# coding: utf-8

import numpy as np
from numpy.core.umath_tests import inner1d
import scipy
import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
import gensim
from collections import Counter
from collections import defaultdict
import heapq
import nltk
import random
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time


MAX_INNER1D_ELEMENTS = 500000000


def set_max_inner1d_elements(value):
    """
    value - максимальное значение объектов p_tdw, которое мы можем одновременно хранить в памяти
    объём потребления памяти равен MAX_INNER1D_ELEMENTS * sizeof(float64)
    """
    global MAX_INNER1D_ELEMENTS
    MAX_INNER1D_ELEMENTS = value


def memory_efficient_inner1d(fst_arr, fst_indices, snd_arr, snd_indices):
    """
    fst_arr
    fst_indices
    snd_arr
    snd_indices
    """
    assert fst_arr.shape[1] == snd_arr.shape[1]
    assert len(fst_indices) == len(snd_indices)

    _, T = fst_arr.shape
    size = len(fst_indices)
    result = np.zeros(size)
    batch_size = MAX_INNER1D_ELEMENTS / T
    
    start = 0
    while start < size:
        finish = min(start + batch_size, size)
        result[start:finish] = inner1d(fst_arr[fst_indices[start:finish], :], snd_arr[snd_indices[start:finish], :])
        start = finish
    
    return result


def create_calculate_likelihood_like_function(n_dw_matrix, loss_function=None):
    if loss_function is None:
        loss_function = LogFunction()

    D, W = n_dw_matrix.shape
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    
    def fun(phi_matrix, theta_matrix):
        s_data = loss_function.calc(memory_efficient_inner1d(theta_matrix, docptr, np.transpose(phi_matrix), wordptr))
        return np.sum(n_dw_matrix.data * s_data)

    return fun


def em_optimization(
    n_dw_matrix, 
    phi_matrix,
    theta_matrix,
    regularization_list,
    iters_count=100,
    loss_function=None,
    iteration_callback=None,
    const_phi=False,
    params=None
):
    if loss_function is None:
        loss_function = LogFunction()
        
    if params is None:
        params = {}
    return_counters = params.get('return_counters', False)

    D, W = n_dw_matrix.shape
    T = phi_matrix.shape[0]
    phi_matrix = np.copy(phi_matrix)
    theta_matrix = np.copy(theta_matrix)
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    
    start_time = time.time()
    for it in xrange(iters_count):
        phi_matrix_tr = np.transpose(phi_matrix)
        # следующая строчка это 60% времени работы алгоритма
        s_data = loss_function.calc_der(memory_efficient_inner1d(theta_matrix, docptr, phi_matrix_tr, wordptr))
        # следующая часть это 25% времени работы алгоритма
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * s_data, 
                n_dw_matrix.indices, 
                n_dw_matrix.indptr
            ), 
            shape=n_dw_matrix.shape
        )
        A_tr = A.tocsc().transpose()
        # Остальное это 15% времени
        n_tw = np.transpose(A_tr.dot(theta_matrix)) * phi_matrix
        n_dt = A.dot(phi_matrix_tr) * theta_matrix
        
        r_tw, r_dt = regularization_list[it](n_tw, n_dt)
        n_tw += r_tw
        n_dt += r_dt
        n_tw[n_tw < 0] = 0
        n_dt[n_dt < 0] = 0
        
        if not const_phi:
            n_tw[np.sum(n_tw, axis=1) < 1e-20, :] += 1
            phi_matrix = n_tw / np.sum(n_tw, axis=1)[:, np.newaxis]

        n_dt[np.sum(n_dt, axis=1) < 1e-20, :] += 1
        theta_matrix = n_dt / np.sum(n_dt, axis=1)[:, np.newaxis]
        
        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)
    
    print 'Iters time', time.time() - start_time
    
    if return_counters:
        return phi_matrix, theta_matrix, n_tw, n_dt
    else:
        return phi_matrix, theta_matrix


def naive_thetaless_em_optimization(
    n_dw_matrix, 
    phi_matrix,
    regularization_list,
    iters_count=100,
    iteration_callback=None,
    theta_matrix=None,
    params=None
):
    if params is None:
        params = {}
    return_counters = params.get('return_counters', False)

    D, W = n_dw_matrix.shape
    T = phi_matrix.shape[0]
    phi_matrix = np.copy(phi_matrix)
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    
    start_time = time.time()
    for it in xrange(iters_count):
        phi_rev_matrix = np.transpose(phi_matrix / np.sum(phi_matrix, axis=0))
        theta_matrix = n_dw_matrix.dot(phi_rev_matrix)
        
        theta_matrix[np.sum(theta_matrix, axis=1) < 1e-20, :] += 1
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]
        phi_matrix_tr = np.transpose(phi_matrix)
        
        s_data = 1. / memory_efficient_inner1d(theta_matrix, docptr, phi_matrix_tr, wordptr)
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data  * s_data , 
                n_dw_matrix.indices, 
                n_dw_matrix.indptr
            ), 
            shape=n_dw_matrix.shape
        ).tocsc()
            
        n_tw = (A.T.dot(theta_matrix)).T * phi_matrix
        r_tw, _ = regularization_list[it](n_tw, theta_matrix)
        n_tw += r_tw
        n_tw[n_tw < 0] = 0
        n_tw[np.sum(n_tw, axis=1) < 1e-20, :] += 1
        phi_matrix = n_tw / np.sum(n_tw, axis=1)[:, np.newaxis]

        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)
    
    print 'Iters time', time.time() - start_time    
    
    if return_counters:
        return phi_matrix, theta_matrix, n_tw, None
    else:
        return phi_matrix, theta_matrix


def artm_thetaless_em_optimization(
    n_dw_matrix, 
    phi_matrix,
    regularization_list,
    iters_count=100,
    iteration_callback=None,
    theta_matrix=None,
    params=None
):
    if params is None:
        params = {}
    use_B_cheat = params.get('use_B_cheat', False)
    return_counters = params.get('return_counters', False)
                             
    D, W = n_dw_matrix.shape
    T = phi_matrix.shape[0]
    phi_matrix = np.copy(phi_matrix)
    docptr = []
    docsizes = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        size = indptr[doc_num + 1] - indptr[doc_num]
        docptr.extend([doc_num] * size)
        if use_B_cheat:
            docsizes.extend([size] * size)
        else:
            docsizes.extend([np.sum(n_dw_matrix.data[indptr[doc_num]:indptr[doc_num + 1]])] * size)
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    docsizes = np.array(docsizes)
    
    B = scipy.sparse.csr_matrix(
        (
            1. * n_dw_matrix.data / docsizes, 
            n_dw_matrix.indices, 
            n_dw_matrix.indptr
        ), 
        shape=n_dw_matrix.shape
    ).tocsc()
    
    start_time = time.time()
    for it in xrange(iters_count):
        word_norm = np.sum(phi_matrix, axis=0)
        word_norm[word_norm == 0] = 1e-20
        phi_rev_matrix = np.transpose(phi_matrix / word_norm)
        
        theta_matrix = n_dw_matrix.dot(phi_rev_matrix)
        theta_matrix[np.sum(theta_matrix, axis=1) < 1e-20, :] += 1
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]
        phi_matrix_tr = np.transpose(phi_matrix)
        
        s_data = 1. / (memory_efficient_inner1d(theta_matrix, docptr, phi_matrix_tr, wordptr) + 1e-20)
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data  * s_data , 
                n_dw_matrix.indices, 
                n_dw_matrix.indptr
            ), 
            shape=n_dw_matrix.shape
        ).tocsc()
            
        n_tw = A.T.dot(theta_matrix).T * phi_matrix
        
        r_tw, r_dt = regularization_list[it](n_tw, theta_matrix)
        theta_indices = theta_matrix > 1e-10
        r_dt[theta_indices] /= theta_matrix[theta_indices]
        r_dt[~theta_indices] = 0.
        
        g_dt = A.dot(phi_matrix_tr) + r_dt
        tmp = g_dt.T * B / word_norm
        r_tw += (tmp - np.einsum('ij,ji->i', phi_rev_matrix, tmp)) * phi_matrix
        
        n_tw += r_tw
        n_tw[n_tw < 1e-20] = 0

        n_tw[np.sum(n_tw, axis=1) < 1e-20, :] += 1
        phi_matrix = n_tw / np.sum(n_tw, axis=1)[:, np.newaxis]

        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)
    
    print 'Iters time', time.time() - start_time    
    
    if return_counters:
        return phi_matrix, theta_matrix, n_tw, None
    else:
        return phi_matrix, theta_matrix


def gradient_optimization(
    n_dw_matrix, 
    phi_matrix,
    theta_matrix,
    regularization_gradient_list,
    iters_count=100,
    loss_function=LogFunction(),
    iteration_callback=None,
    learning_rate=1.,
    params=None
):
    if params is None:
        params = {}
    return_counters = params.get('return_counters', False)

    D, W = n_dw_matrix.shape
    T = phi_matrix.shape[0]
    phi_matrix = np.copy(phi_matrix)
    theta_matrix = np.copy(theta_matrix)
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    
    start_time = time.time()
    for it in xrange(iters_count):
        phi_matrix_tr = np.transpose(phi_matrix)
        # следующая строчка это 60% времени работы алгоритма
        s_data = loss_function.calc_der(memory_efficient_inner1d(theta_matrix, docptr, phi_matrix_tr, wordptr))
        # следующая часть это 25% времени работы алгоритма
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * s_data, 
                n_dw_matrix.indices, 
                n_dw_matrix.indptr
            ), 
            shape=n_dw_matrix.shape
        ).tocsc()
        # Остальное это 15% времени
        g_tw = theta_matrix.T * A
        g_dt = A.dot(phi_matrix_tr)
        
        r_tw, r_dt = regularization_gradient_list[it](phi_matrix, theta_matrix)
        g_tw += r_tw
        g_dt += r_dt
        
        g_tw -= np.sum(g_tw * phi_matrix, axis=1)[:, np.newaxis]
        g_dt -= np.sum(g_dt * theta_matrix, axis=1)[:, np.newaxis]
        
        phi_matrix += g_tw * learning_rate
        theta_matrix += g_dt * learning_rate
        
        phi_matrix[phi_matrix < 0] = 0
        theta_matrix[theta_matrix < 0] = 0
        
        phi_matrix[np.sum(phi_matrix, axis=1) < 1e-20, :] += 1
        phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]
        
        theta_matrix[np.sum(theta_matrix, axis=1) < 1e-20, :] += 1
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]
        
        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)
    
    print 'Iters time', time.time() - start_time  
    
    if return_counters:
        return phi_matrix, theta_matrix, n_tw, n_dt
    else:
        return phi_matrix, theta_matrix


def svm_score(theta, targets, verbose=True):
    C_2d_range = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    gamma_2d_range = [1e-3, 1e-2, 1e-1, 1, 1e1]
    best_C, best_gamma, best_val = None, None, 0.
    best_cv_algo_score_on_test = 0.
    X_train, X_test, y_train, y_test = train_test_split(theta, targets, test_size=0.30, stratify=targets, random_state=42)
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            val = np.mean(cross_val_score(SVC(C=C, gamma=gamma), X_train, y_train, scoring='accuracy', cv=4))
            algo = SVC(C=C, gamma=gamma).fit(X_train, y_train)
            test_score = accuracy_score(y_test, algo.predict(X_test))
            if verbose:
                print 'SVM(C={}, gamma={}) cv-score: {}  test-score: {}'.format(
                    C,
                    gamma,
                    round(val, 3),
                    round(test_score, 3)
                )
            if val > best_val:
                best_val = val
                best_C = C
                best_gamma = gamma
                best_cv_algo_score_on_test = test_score
    if verbose:
        print '\n\n\nBest cv params: C={}, gamma={}\nCV score: {}\nTest score:{}'.format(
            best_C,
            best_gamma,
            round(best_val, 3),
            round(best_cv_algo_score_on_test, 3)
        )
    return best_C, best_gamma, best_val, best_cv_algo_score_on_test


def artm_calc_topic_correlation(phi):
    T, W = phi.shape
    return (np.sum(np.sum(phi, axis=0) ** 2) - np.sum(phi ** 2)) / (T * (T - 1))


def artm_get_kernels(phi):
    T, W = phi.shape
    return [
        set(np.where(phi[t, :] * W > 1)[0])
        for t in xrange(T)
    ]


def artm_get_kernels_sizes(phi):
    return [len(kernel) for kernel in artm_get_kernels(phi)]


def artm_get_avg_pairwise_kernels_jacards(phi):
    T, W = phi.shape
    kernels = artm_get_kernels(phi)
    res = 0.
    for i in xrange(T):
        for j in xrange(T):
            if i != j:
                res += 1. * len(kernels[i] & kernels[j]) / (len(kernels[i] | kernels[j]) + 0.1)
    return res / T / (T - 1)


def artm_get_avg_top_words_jacards(phi, top_size):
    T, W = phi.shape
    top_words = [
        set(heapq.nlargest(top_size, xrange(W), key=lambda w: phi[t, w]))
        for t in xrange(T)
    ]
    res = 0.
    for i in xrange(T):
        for j in xrange(T):
            if i != j:
                res += 1. * len(top_words[i] & top_words[j]) / (len(top_words[i] | top_words[j]) + 0.1)
    return res / T / (T - 1)


def artm_calc_phi_uniqueness_measures(phi):
    T, W = phi.shape
    res = []
    nres = []
    for t in xrange(T):
        positions = phi[t, :] == 0.
        topics = [x for x in xrange(T) if x != t]
        if np.sum(positions) == 0:
            res.append(0.)
            nres.append(0.)
        else:
            rank = np.linalg.matrix_rank(phi[np.ix_(topics, positions)])
            if rank == T - 1:
                max_val = np.min(np.linalg.svd(phi[topics, :])[1])
                curr_val = np.min(np.linalg.svd(phi[np.ix_(topics, positions)])[1])
                res.append(curr_val)
                nres.append(curr_val / max_val)
            else:
                res.append(0.)
                nres.append(0.)
    return res, nres


def artm_calc_perplexity_factory(n_dw_matrix):
    helper = create_calculate_likelihood_like_function(
        loss_function=LogFunction(),
        n_dw_matrix=n_dw_matrix
    )
    total_words_number = n_dw_matrix.sum()
    return lambda phi, theta: np.exp(- helper(phi, theta) / total_words_number)     


def artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, documents_number, top_size):
    def fun(phi):
        T, W = phi.shape
        pmi = 0.
        for t in xrange(T):
            top = heapq.nlargest(top_size, xrange(W), key=lambda w: phi[t, w])
            for w1 in top:
                for w2 in top:
                    if w1 != w2:
                        pmi += np.log(documents_number * (doc_cooccurences.get((w1, w2), 0.) + 1e-4) * 1. / doc_occurences.get(w1, 0) / doc_occurences.get(w2, 0))
        return pmi / (T * top_size * (top_size - 1))
    return fun

