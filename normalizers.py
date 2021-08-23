#!/bin/python3
from collections import defaultdict
import torch
from torch import linalg as LA


def normed(x):
    return x/x.norm()

class LipschitzNormalizer(object):

    def __init__(self, params, niter=1):
        self.params = list(params)
        # uv stores for each parameter a tuple (u,v) of the singular vectors
        self.uv = defaultdict(tuple)
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

    def eval_lipnrm(self):

        nrm = 1
        for p in self.params:
            if p.ndim == 2:
                nrm *= LA.norm(p, ord=2)
        return nrm





