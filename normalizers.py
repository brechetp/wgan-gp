#!/bin/python3
from collections import defaultdict
import torch
from torch import linalg as LA
from models import DeepCNNDiscriminator
import torch.nn.functional as F
import torch.nn as nn


def normed(x):
    return x/x.norm()

class LipschitzNormalizer(nn.Module):

    def __init__(self, module, niter=1):
        self.module = module # will be e.g. a Conv2d
        self.niter = 1
        # for module in net.modules():
            # if isinstance(module, nn.Conv2d):  # have to record the sizes
                # for p in module.params():
                    # self.sz[p] =
        # uv stores for each parameter a tuple (u,v) of the singular vectors
        self.uv = defaultdict(tuple)
        self.sz = defaultdict(tuple)
        if isinstance(net, DeepCNNDiscriminator):
            self.sz = net.sz  # the input / output size of the layers
        # self.sz = defaultdict(tuple)  # for the conv sizes
        self.niter = niter

    def normalize(self):
        # for all the parameters
        for p in self.params:
            if p.ndim == 2:
                (u, v) = self.uv.get(p, (torch.randn(p.size(0), 1, device=p.device), torch.randn(p.size(1), 1, device=p.device)))
                for n in range(self.niter):
                    v = normed(p.T.mm(u))
                    u = normed(p.mm(v))
                p.data.div_( (u.T.mm( p ).mm( v)).item())
                self.uv[p] = (u,v)
            elif p.ndim == 4:  # convolution
                inputsize, outputsize, stride, padding = self.sz[p]
                (u, v) = self.uv.get(p, (torch.randn(outputsize, device=p.device), torch.randn(inputsize, device=p.device)))
                for n in range(self.niter):
                    # Perform the (transpose) convolution
                    vv = F.conv_transpose2d(u, p, stride=stride, padding=padding)
                    v = normed(vv)
                    uu = F.conv2d(v, p, stride=stride, padding=padding)
                    u = normed(uu)
                nrm = (F.conv2d(v, p, stride=stride, padding=padding) * u).sum()
                p.data.div_(nrm)  # modify W contents
                self.uv[p] = (u,v)

    def initialize(self):
        pass
        # initialize the sizes in th case of convolution networks
    def eval_lipnrm(self):

        nrm = 1
        for p in self.params:
            if p.ndim == 2:
                nrm *= LA.norm(p, ord=2)
        return nrm

class LipschitzNormalizer(object):

    def __init__(self, net, niter=1):
        self.params = list(net.parameters())
        # for module in net.modules():
            # if isinstance(module, nn.Conv2d):  # have to record the sizes
                # for p in module.params():
                    # self.sz[p] =
        # uv stores for each parameter a tuple (u,v) of the singular vectors
        self.uv = defaultdict(tuple)
        self.sz = defaultdict(tuple)
        if isinstance(net, DeepCNNDiscriminator):
            self.sz = net.sz  # the input / output size of the layers
        # self.sz = defaultdict(tuple)  # for the conv sizes
        self.niter = niter

    def normalize(self):
        # for all the parameters
        for p in self.params:
            if p.ndim == 2:
                (u, v) = self.uv.get(p, (torch.randn(p.size(0), 1, device=p.device), torch.randn(p.size(1), 1, device=p.device)))
                for n in range(self.niter):
                    v = normed(p.T.mm(u))
                    u = normed(p.mm(v))
                p.data.div_( (u.T.mm( p ).mm( v)).item())
                self.uv[p] = (u,v)
            elif p.ndim == 4:  # convolution
                inputsize, outputsize, stride, padding = self.sz[p]
                (u, v) = self.uv.get(p, (torch.randn(outputsize, device=p.device), torch.randn(inputsize, device=p.device)))
                for n in range(self.niter):
                    # Perform the (transpose) convolution
                    vv = F.conv_transpose2d(u, p, stride=stride, padding=padding)
                    v = normed(vv)
                    uu = F.conv2d(v, p, stride=stride, padding=padding)
                    u = normed(uu)
                nrm = (F.conv2d(v, p, stride=stride, padding=padding) * u).sum()
                p.data.div_(nrm)  # modify W contents
                self.uv[p] = (u,v)

    def initialize(self):
        pass
        # initialize the sizes in th case of convolution networks
    def eval_lipnrm(self):

        nrm = 1
        for p in self.params:
            if p.ndim == 2:
                nrm *= LA.norm(p, ord=2)
        return nrm





