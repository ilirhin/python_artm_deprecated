# coding: utf-8

import numpy as np
from numpy.core.umath_tests import inner1d
from artm_utils.loss_functions import LogFunction
import heapq
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import scipy.sparse


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


def get_docptr(n_dw_matrix):
    D, W = n_dw_matrix.shape
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    return np.array(docptr)


def create_calculate_likelihood_like_function(n_dw_matrix, loss_function=None):
    if loss_function is None:
        loss_function = LogFunction()

    docptr = get_docptr(n_dw_matrix)
    wordptr = n_dw_matrix.indices
    
    def fun(phi_matrix, theta_matrix):
        s_data = loss_function.calc(memory_efficient_inner1d(theta_matrix, docptr, np.transpose(phi_matrix), wordptr))
        return np.sum(n_dw_matrix.data * s_data)

    return fun


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


def pairwise_counters_2_sparse_matrix(cooccurences):
    row = []
    col = []
    data = []
    for (w1, w2), value in cooccurences.iteritems():
        row.append(w1)
        col.append(w2)
        data.append(value)
    return scipy.sparse.csr_matrix((data, (row, col)))
               

def artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, documents_number, top_size):
    def fun(phi):
        T, W = phi.shape
        pmi = 0.
        additive_smooth_constant = 1. / documents_number
        for t in xrange(T):
            #top = heapq.nlargest(top_size, xrange(W), key=lambda w: phi[t, w])
            top = np.argpartition(phi[t, :], -top_size)[-top_size:]
            cooccurences = doc_cooccurences[top, :][:, top].todense()
            occurences = doc_occurences[top]
            values = np.log((cooccurences + additive_smooth_constant) * documents_number / occurences[:, np.newaxis] / occurences[np.newaxis, :])
            pmi += values.sum() - values[np.diag_indices(len(values))].sum()
        return pmi / (T * top_size * (top_size - 1))
    return fun


def artm_calc_positive_pmi_top_factory(doc_occurences, doc_cooccurences, documents_number, top_size):
    def fun(phi):
        T, W = phi.shape
        pmi = 0.
        additive_smooth_constant = 1. / documents_number
        for t in xrange(T):
            top = np.argpartition(phi[t, :], -top_size)[-top_size:]
            cooccurences = doc_cooccurences[top, :][:, top].todense()
            occurences = doc_occurences[top]
            values = np.log((cooccurences + additive_smooth_constant) * documents_number / occurences[:, np.newaxis] / occurences[np.newaxis, :])
            values[values < 0.] = 0.
            pmi += values.sum() - values[np.diag_indices(len(values))].sum()
        return pmi / (T * top_size * (top_size - 1))
    return fun

