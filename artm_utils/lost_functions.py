import numpy as np


class LogFunction(object):
    def calc(self, x):
        return np.log(x + 1e-20)
    def calc_der(self, x):
        return 1. / (x + 1e-20)
    

class IdFunction(object):
    def calc(self, x):
        return x + 1e-20
    def calc_der(self, x):
        return np.ones_like(x)
    

class SquareFunction(object):
    def calc(self, x):
        return (x + 1e-20) ** 2
    def calc_der(self, x):
        return 2. * (x + 1e-20) ** 2
    

class CubeLogFunction(object):
    def calc(self, x):
        return np.log(x + 1e-20) ** 3
    def calc_der(self, x):
        return 3. * np.log(x + 1e-20) ** 2 / (x + 1e-20)
    

class SquareLogFunction(object):
    def calc(self, x):
        return np.log(x + 1e-20) * np.abs(np.log(x + 1e-20))
    def calc_der(self, x):
        return 2. * np.abs(np.log(x + 1e-20)) / (x + 1e-20)

    
class FiveLogFunction(object):
    def calc(self, x):
        return np.log(x + 1e-20) ** 5
    def calc_der(self, x):
        return 5. * np.log(x + 1e-20) ** 4 / (x + 1e-20)
    

class CubeRootLogFunction(object):
    def calc(self, x):
        return np.cbrt(np.log(x + 1e-20))
    def calc_der(self, x):
        return 1. / 3 / (np.cbrt(np.log(x + 1e-20)) ** 2) / (x + 1e-20)
    
    
class SquareRootLogFunction(object):
    def calc(self, x):
        return np.sqrt(- np.log(x + 1e-20))
    def calc_der(self, x):
        return 1. / 2. / np.sqrt(- np.log(x + 1e-20)) / (x + 1e-20)
    

class ExpFunction(object):
    def calc(self, x):
        return np.exp(x)
    def calc_der(self, x):
        return np.exp(x)

    
class EntropyFunction(object):
    def calc(self, x):
        return (np.log(x + 1e-20) + 50.) * (x + 1e-20)
    def calc_der(self, x):
        return np.log(x + 1e-20) + 50.
