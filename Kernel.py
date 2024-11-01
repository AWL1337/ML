import math

import numpy as np


def get_kernel(kernel_type):
    if kernel_type == 'Gaussian':
        return GaussianKernel()
    if kernel_type == 'Uniform':
        return UniformKernel()
    if kernel_type.startswith('General_'):
        a, b = map(int, kernel_type.split('_')[1:])
        return GeneralKernel(a, b)


class Kernel:
    def __init__(self, name):
        self.name = name


class GaussianKernel(Kernel):
    def __init__(self):
        super().__init__("GaussianKernel")

    def get_name(self):
        return super().name

    def process(self, data):
        return 1/math.sqrt(2*math.pi) * np.exp(-0.5*np.array(data)**2)


class UniformKernel(Kernel):
    def __init__(self):
        super().__init__("UniformKernel")

    def get_name(self):
        return super().name

    def process(self, data):
        return np.ones_like(data)


class GeneralKernel(Kernel):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        super().__init__("GeneralKernel a:{} b:{}".format(a, b))

    def get_name(self):
        return super().name

    def process(self, data):
        return (1 - np.abs(data) ** self.a) ** self.b