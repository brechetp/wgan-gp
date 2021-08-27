
import torch
import torch.nn as nn
from .discriminator import Discriminator
import numpy as np
import math
from collections import defaultdict
from torch.nn.utils.parametrizations import spectral_norm

def prod(l, s=1):
    for i in range(len(l)):
        s*=l[i]
    return s


class  TwoLayerNet(nn.Module):

    def __init__(self, din, dhid, sout=(3, 32, 32), activation=nn.ELU):

        super().__init__()
        layers = []
        dout  = prod(sout)
        self.sout = sout
        self.sri = 1/math.sqrt(dhid)
        self.net = nn.Sequential(spectral_norm(nn.Linear(din, dhid)), activation(), spectral_norm(nn.Linear(dhid, dout)))
        self.tanh = nn.Tanh()

        nn.init.normal_(self.net[0].weight, 0.0, std=1/din)
        nn.init.uniform_(self.net[2].weight, -1/math.sqrt(dout), 1/math.sqrt(dout))
        return

    def forward(self, x):

        x = self.net(x)
        x = x * self.sri
        x = self.tanh(x)
        x = x.view(x.size(0), *self.sout)
        return x

    def compute_svdvals(self):
        """Return the values of the SVD from the weights matrix"""
        return torch.linalg.svdvals(self.net[0].weight.data).detach().cpu().numpy(), torch.linalg.svdvals(self.net[2].weight.data).detach().cpu().numpy()


class DeepCNNDiscriminator(Discriminator):

    def __init__(self, sin=(3, 32, 32), alpha=1., n_filters=32, activation=nn.ELU, ln=1):

        super().__init__()

        kernels = [n_filters*(2**i) for i in np.repeat(range(3), 2)] #+ [n_filters*4] # e.g. [32,32, 64, 64, 128, 128]
        # kernels = [n_filters*(2**i) for i in range(6)]  # e.g. [32, 64, 128, 256, 512, 1024]
        strides = [*np.tile(range(2, 0, -1), 3), 1]  # alternating [2, 1, 2, 1, 2, 1]
        # strides = 5*[2] + [1]
        # ksz = 5*[4] + [1]
        ksz = [4 if s == 2 else 3 for s in strides]
        # pds = 5*[1] + [0]
        pds = 6*[1] #+ [0]
        h = sin[1]  # assume square imaages
        cout = sin[0] # for the initialization
        layers = []
        self.inputsize = []
        # self.sz = defaultdict(tuple)
        for idx in range(len(kernels)):
            cin, cout = cout, kernels[idx]
            inputsize = (1, cin, h, h)  # 1 is for the batch dimension
            h = math.floor((h + 2 * pds[idx] - 1 * (ksz[idx] -1) - 1) / strides[idx] + 1)
            outputsize = (1, cout, h, h)
            layers.append(spectral_norm(nn.Conv2d(cin, cout, ksz[idx], strides[idx], pds[idx]), n_power_iterations=ln, inputsize=inputsize, outputsize=outputsize))
            layers.append(activation())
            # p = layers[-2].weight  # the weight parameter
            # self.inputsize[p] = (inputsize, outputsize, strides[idx], 1)  # record the size of the input and output
            self.inputsize.append(inputsize)

        self.conv = nn.Sequential(*layers)
        self.dense = spectral_norm(nn.Linear(h*h*cout ,1))
        self.alpha = alpha

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.alpha * x
        return x

    def compute_svdvals(self):

        Vs = []
        idx = 0
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                if layer.stride[0] == 1:
                    transform = torch.fft.fft2(layer.weight.data, s=self.inputsize[idx][-2:])
                    Vs.append((idx, torch.linalg.svdvals(transform).detach().cpu().numpy()))
                idx += 1
        Vs.append((idx, torch.linalg.svdvals(self.dense.weight.data).detach().cpu().numpy()))
        return Vs




