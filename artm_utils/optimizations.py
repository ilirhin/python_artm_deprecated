# coding: utf-8

import numpy as np
import scipy.sparse
from artm_utils.loss_functions import LogFunction
from artm_utils.calculations import get_docptr
from artm_utils.calculations import memory_efficient_inner1d
import time

"""
Обозначения:
    s_dw = sum_t phi_tw * theta_dt
    A_dw = n_dw / s_dw
    B_dw = n_dw / f_d, где f_d либо sum_w n_dw, либо sum_w [n_dw > 0]
"""

OPT_EPS = 1e-20


def get_prob_matrix_by_counters(counters):
    res = np.copy(counters)
    res[res < 0] = 0
    res[np.sum(res, axis=1) < OPT_EPS, :] += 1
    res /= np.sum(res, axis=1)[:, np.newaxis]
    return res


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

    phi_matrix = np.copy(phi_matrix)
    theta_matrix = np.copy(theta_matrix)

    docptr = get_docptr(n_dw_matrix)
    wordptr = n_dw_matrix.indices

    start_time = time.time()
    for it in xrange(iters_count):
        phi_matrix_tr = np.transpose(phi_matrix)
        s_data = loss_function.calc_der(memory_efficient_inner1d(theta_matrix, docptr, phi_matrix_tr, wordptr))

        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * s_data,
                n_dw_matrix.indices,
                n_dw_matrix.indptr
            ),
            shape=n_dw_matrix.shape
        )
        n_dt = A.dot(phi_matrix_tr) * theta_matrix
        n_tw = np.transpose(A.tocsc().transpose().dot(theta_matrix)) * phi_matrix

        r_tw, r_dt = regularization_list[it](phi_matrix, theta_matrix, n_tw, n_dt)
        n_tw += r_tw
        n_dt += r_dt

        if not const_phi:
            phi_matrix = get_prob_matrix_by_counters(n_tw)
        theta_matrix = get_prob_matrix_by_counters(n_dt)

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
    loss_function = LogFunction()
    return_counters = params.get('return_counters', False)

    phi_matrix = np.copy(phi_matrix)

    docptr = get_docptr(n_dw_matrix)
    wordptr = n_dw_matrix.indices

    start_time = time.time()
    for it in xrange(iters_count):
        phi_rev_matrix = np.transpose(phi_matrix / np.sum(phi_matrix, axis=0))
        phi_matrix_tr = np.transpose(phi_matrix)
        theta_matrix = get_prob_matrix_by_counters(n_dw_matrix.dot(phi_rev_matrix))

        s_data = loss_function.calc_der(memory_efficient_inner1d(theta_matrix, docptr, phi_matrix_tr, wordptr))
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * s_data,
                n_dw_matrix.indices,
                n_dw_matrix.indptr
            ),
            shape=n_dw_matrix.shape
        ).tocsc()

        n_tw = (A.T.dot(theta_matrix)).T * phi_matrix
        r_tw, _ = regularization_list[it](phi_matrix, theta_matrix, n_tw, theta_matrix)
        n_tw += r_tw
        phi_matrix = get_prob_matrix_by_counters(n_tw)

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
    loss_function = LogFunction()
    use_B_cheat = params.get('use_B_cheat', False)
    return_counters = params.get('return_counters', False)

    phi_matrix = np.copy(phi_matrix)

    docptr = get_docptr(n_dw_matrix)
    wordptr = n_dw_matrix.indices

    D, _ = n_dw_matrix.shape
    docsizes = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        size = indptr[doc_num + 1] - indptr[doc_num]
        if use_B_cheat:
            docsizes.extend([size] * size)
        else:
            docsizes.extend([np.sum(n_dw_matrix.data[indptr[doc_num]:indptr[doc_num + 1]])] * size)
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
        phi_matrix_tr = np.transpose(phi_matrix)
        phi_rev_matrix = get_prob_matrix_by_counters(phi_matrix_tr)
        theta_matrix = get_prob_matrix_by_counters(n_dw_matrix.dot(phi_rev_matrix))

        s_data = loss_function.calc_der(memory_efficient_inner1d(theta_matrix, docptr, phi_matrix_tr, wordptr))
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * s_data,
                n_dw_matrix.indices,
                n_dw_matrix.indptr
            ),
            shape=n_dw_matrix.shape
        ).tocsc()

        n_tw = A.T.dot(theta_matrix).T * phi_matrix
        r_tw, r_dt = regularization_list[it](phi_matrix, theta_matrix, n_tw, theta_matrix)

        theta_indices = theta_matrix > OPT_EPS
        r_dt[theta_indices] /= theta_matrix[theta_indices]
        r_dt[~theta_indices] = 0.

        g_dt = A.dot(phi_matrix_tr) + r_dt
        tmp = g_dt.T * B / (phi_matrix_tr.sum(axis=1) + OPT_EPS)
        r_tw += (tmp - np.einsum('ij,ji->i', phi_rev_matrix, tmp)) * phi_matrix

        n_tw += r_tw
        phi_matrix = get_prob_matrix_by_counters(n_tw)

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
        loss_function=None,
        iteration_callback=None,
        learning_rate=1.,
):
    if loss_function is None:
        loss_function = LogFunction()

    phi_matrix = np.copy(phi_matrix)
    theta_matrix = np.copy(theta_matrix)

    docptr = get_docptr(n_dw_matrix)
    wordptr = n_dw_matrix.indices

    start_time = time.time()
    for it in xrange(iters_count):
        phi_matrix_tr = np.transpose(phi_matrix)
        s_data = loss_function.calc_der(memory_efficient_inner1d(theta_matrix, docptr, phi_matrix_tr, wordptr))
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * s_data,
                n_dw_matrix.indices,
                n_dw_matrix.indptr
            ),
            shape=n_dw_matrix.shape
        ).tocsc()
        g_tw = theta_matrix.T * A
        g_dt = A.dot(phi_matrix_tr)

        r_tw, r_dt = regularization_gradient_list[it](phi_matrix, theta_matrix, phi_matrix, theta_matrix)
        g_tw += r_tw
        g_dt += r_dt

        g_tw -= np.sum(g_tw * phi_matrix, axis=1)[:, np.newaxis]
        g_dt -= np.sum(g_dt * theta_matrix, axis=1)[:, np.newaxis]

        phi_matrix += g_tw * learning_rate
        theta_matrix += g_dt * learning_rate

        phi_matrix = get_prob_matrix_by_counters(phi_matrix)
        theta_matrix = get_prob_matrix_by_counters(theta_matrix)

        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)

    print 'Iters time', time.time() - start_time

    return phi_matrix, theta_matrix
