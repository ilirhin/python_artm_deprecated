# coding: utf-8

import numpy as np


class LogFunction(object):
    @staticmethod
    def calc(x):
        return np.log(x + 1e-20)

    @staticmethod
    def calc_der(x):
        return 1. / (x + 1e-20)


class IdFunction(object):
    @staticmethod
    def calc(x):
        return x + 1e-20

    @staticmethod
    def calc_der(x):
        return np.ones_like(x)


class SquareFunction(object):
    @staticmethod
    def calc(x):
        return (x + 1e-20) ** 2

    @staticmethod
    def calc_der(x):
        return 2. * (x + 1e-20) ** 2


class CubeLogFunction(object):
    @staticmethod
    def calc(x):
        return np.log(x + 1e-20) ** 3

    @staticmethod
    def calc_der(x):
        return 3. * np.log(x + 1e-20) ** 2 / (x + 1e-20)


class SquareLogFunction(object):
    @staticmethod
    def calc(x):
        return np.log(x + 1e-20) * np.abs(np.log(x + 1e-20))

    @staticmethod
    def calc_der(x):
        return 2. * np.abs(np.log(x + 1e-20)) / (x + 1e-20)


class FiveLogFunction(object):
    @staticmethod
    def calc(x):
        return np.log(x + 1e-20) ** 5

    @staticmethod
    def calc_der(x):
        return 5. * np.log(x + 1e-20) ** 4 / (x + 1e-20)


class CubeRootLogFunction(object):
    @staticmethod
    def calc(x):
        return np.cbrt(np.log(x + 1e-20))

    @staticmethod
    def calc_der(x):
        return 1. / 3 / (np.cbrt(np.log(x + 1e-20)) ** 2) / (x + 1e-20)


class SquareRootLogFunction(object):
    @staticmethod
    def calc(x):
        return np.sqrt(- np.log(x + 1e-20))

    @staticmethod
    def calc_der(x):
        return 1. / 2. / np.sqrt(- np.log(x + 1e-20)) / (x + 1e-20)


class ExpFunction(object):
    @staticmethod
    def calc(x):
        return np.exp(x)

    @staticmethod
    def calc_der(x):
        return np.exp(x)


class EntropyFunction(object):
    @staticmethod
    def calc(x):
        return (np.log(x + 1e-20) + 50.) * (x + 1e-20)

    @staticmethod
    def calc_der(x):
        return np.log(x + 1e-20) + 50.
