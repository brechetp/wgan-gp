import torch
import torch.nn as nn
from .discriminator import Discriminator

"""
A Linear network
"""
def LinearNet(indim, shape, outdim=1):

    d = [indim] + shape + [outdim]
    depth = len(d) - 1
    netlst = [nn.Linear(d[i], d[i+1]) for i in range(depth)]
    return nn.Sequential(*netlst)

class LinearDiscriminator(Discriminator):

    def __init__(self, indim, shape):
        super().__init__()
        self.main = LinearNet(indim, shape, 1)

    def forward(self, x):
        return self.main(x)

# def LinearGenerator(indim, shape, outdim):

    # d = [indim] + shape + [outdim]
    # depth = len(d) - 1
    # netlst = [nn.Linear(d[i], d[i+1]) for i in range(depth)]
    # return nn.Sequential(netlst)
# class LinearDiscriminator:

    # def __init__(self, dim, shape::list):

        # nn.Dense()
        # return nn.Sequential


    # def forward(self, x):

        # return self.main(x)

