#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable, grad
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import json
import csv
import shutil
import sys
from normalizers import LipschitzNormalizer
import random
import pandas as pd
import matplotlib as plt
# from datetime import datetime
import datetime

import models
import utils
from plot import plot


parser = argparse.ArgumentParser()
parser.add_argument('--output', default="results")
parser.add_argument('--name', default="")
parser.add_argument('--cuda', action='store_true')
parser.add_argument('-bs' ,'--batch-size', default=64, type=int)
parser.add_argument('--num-iter', default=100_000, type=int)
parser.add_argument('-lrd', '--learning-rate-dsc', default=2e-4, type=float)
parser.add_argument('-lrg', '--learning-rate-gen', default=2e-5, type=float)
# parser.add_argument('-b1' ,'--beta1', default=0.5, type=float)
# parser.add_argument('-b2' ,'--beta2', default=0.9, type=float)
# parser.add_argument('-ema', default=0.9999, type=float)
parser.add_argument('-nz' ,'--num-latent', default=128, type=int)
parser.add_argument('-nh', '--num-hidden', default=4000, type=int)
parser.add_argument('-nfd' ,'--num-filters-dsc', default=128, type=int)
parser.add_argument('-nfg' ,'--num-filters-gen', default=128, type=int)
parser.add_argument('-gp', '--gradient-penalty', default=10, type=float)
parser.add_argument('-ln', '--lipschitz-normalizer', default=1, type=int, help='the number of iterations for lipschitz normalizer')
parser.add_argument('-a', '--alpha', default=1., type=float, help='alpha parameter for discriminator')
parser.add_argument('-b', '--beta', default=1., type=float, help='beta parameter for the regularization stength')
# parser.add_argument('-m', '--mode', choices=('gan','ns-gan', 'wgan'), default='wgan')
# parser.add_argument('-c', '--clip', default=0.01, type=float)
parser.add_argument('-d', '--distribution', choices=('normal', 'uniform'), default='normal')
parser.add_argument('--checkpoint', help='checkpoint computed previously')
# parser.add_argument('--batchnorm-dsc', action='store_true')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--inception-score', action='store_true')
parser.add_argument('--default', action='store_true')
parser.add_argument('-u', '--update-frequency', default=3, type=int)
parser.add_argument('-gpu', "--gpu-index", type=int, help="gpu index to use")
parser.add_argument('--pick_random', action="store_true")

args = parser.parse_args()


if args.checkpoint is not None:  # we have some networks weights to continue
    try:
        num_iter = args.num_iter
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        args.__dict__.update(checkpoint["args"].__dict__)
        args.num_iter = num_iter
        df = checkpoint['df'].dropna(how="all")
        if 'niter' in checkpoint.keys():
            ep = checkpoint['niter']['ep']
            n_gen_update = checkpoint['niter']['n_gen_update']
            n_dsc_update = checkpoint['niter']['n_dsc_update']
        else:
            ep = len(checkpoint['df'])
            n_gen_update = int(df.loc[ep, 'niter gen'])
            n_dsc_update = n_gen_update * args.update_frequency



    except RuntimeError as e:
        print('Error loading the checkpoint at {}'.format(e))

else:
    checkpoint = dict()
    ep = 0
    n_gen_update = 0
    n_dsc_update = 0
    names=['set', 'stat', 'net']
    #tries = np.arange(args.ntry)
    sets = ['train']
    stats=['loss']
    nets = ['gen', 'dsc']
    #layers = ['last', 'hidden']
    # columns=pd.MultiIndex.from_product([sets, stats, nets], names=names)
    columns=pd.Index(['niter gen', 'loss gen', 'loss dsc', 'GP', 'IS'])
    df = pd.DataFrame(columns=columns, index=pd.Index([], name='ep'))

    if args.pick_random:
        args.alpha, args.beta, args.num_hidden, args.learning_rate_dsc = utils.pick_random()
        args.learning_rate_gen = args.learning_rate_dsc

dtype = torch.float
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    gpu_index = args.gpu_index % num_gpus if args.gpu_index is not None else random.choice(range(num_gpus))
else:
    gpu_index = 0
# gpu_index = random.choice(range(num_gpus)) if num_gpus > 0  else 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', gpu_index)


start_n_gen = prev_n_gen = n_gen_update

GRADIENT_PENALTY = args.gradient_penalty
OUTPUT_PATH = os.path.join(args.output, args.name)
TENSORBOARD_FLAG = args.tensorboard
INCEPTION_SCORE_FLAG = args.inception_score
UPDATE_FREQUENCY = args.update_frequency
LIPSCHITZ_NORMALIZER = args.lipschitz_normalizer



if args.default:
    try:
        if args.gradient_penalty == 0:
            config = "config/default_%s_wgan_sgd%i_.json"%(args.model, UPDATE_FREQUENCY)
        else:
            config = "config/default_%s_wgangp_sgd%i.json"%(args.model, UPDATE_FREQUENCY)
    except:
        raise ValueError("Not default config available for this.")

    with open(config) as f:
        data = json.load(f)
    args = argparse.Namespace(**data)


BATCH_SIZE = args.batch_size
N_ITER = args.num_iter
LEARNING_RATE_G = args.learning_rate_gen # It is really important to set different learning rates for the discriminator and generator
LEARNING_RATE_D = args.learning_rate_dsc
# BETA_1 = args.beta1
# BETA_2 = args.beta2
# BETA_EMA = args.ema
N_LATENT = args.num_latent
N_FILTERS_G = args.num_filters_gen
N_FILTERS_D = args.num_filters_dsc
# MODE = args.mode
MODE="wgan"
# CLIP = args.clip
DISTRIBUTION = args.distribution
BATCH_NORM_G = True
BATCH_NORM_D = False #args.batchnorm_dsc
N_SAMPLES = 50000
RESOLUTION = 32
N_CHANNEL = 3
START_EPOCH = 0
EVAL_FREQ = 10000
# SEED = args.seed
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# if haskey(checkpoint, "stats"):
total_time = 0
if "total_time" in checkpoint.keys():
    total_time = checkpoint["total_time"]


suffix = ''
if LIPSCHITZ_NORMALIZER:
    suffix += '-lipnrm'
if GRADIENT_PENALTY:
    suffix += '-gp'

NIN = args.num_latent
NHID = args.num_hidden
MODELSTR = os.path.join(f"nin-{NIN}-nhid-{NHID}", f"a-{args.alpha:.1e}-b-{args.beta:.1e}-ln-{LIPSCHITZ_NORMALIZER}")


if LEARNING_RATE_D == LEARNING_RATE_G:
    LRSTR =  f"lr={LEARNING_RATE_D:.1e}"
else:
    LRSTR = 'lrd=%.1e_lrg=%.1e'%(LEARNING_RATE_D, LEARNING_RATE_G)

OUTPUT_PATH = os.path.join(OUTPUT_PATH, MODELSTR, LRSTR)

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "plots"), exist_ok=True)

if TENSORBOARD_FLAG:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_PATH, 'tensorboard'))
    writer.add_text('config', json.dumps(vars(args), indent=2, sort_keys=True))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True)

print('Init....')
if not os.path.exists(os.path.join(OUTPUT_PATH, 'checkpoints')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'checkpoints'))
if not os.path.exists(os.path.join(OUTPUT_PATH, 'gen')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'gen'))

resf = open(os.path.join(OUTPUT_PATH, 'results.csv'), 'a')
logf = open(os.path.join(OUTPUT_PATH, 'logs.txt'), 'a')

print("start at ", datetime.datetime.now(), file=logf)


torch.autograd.set_detect_anomaly(True)

# start_iter = 0
# if 'df' in checkpoint.keys():
    # start_iter = len(checkpoint['df'])

# dtypes = np.dtype([("niter gen", int), ("loss gen", float), ("loss dsc", float), ("GP", float), ("IS", float)])
gen_upd_per_epoch = len(trainloader) // UPDATE_FREQUENCY
num_ep = args.num_iter // (gen_upd_per_epoch)
print(f"ep: {ep}, gen_upd_per_epoch: {gen_upd_per_epoch}, num_ep: {num_ep}", file=logf, flush=True)
df = df.reindex(df.index.append(pd.RangeIndex(ep +1, ep+num_ep+1)))  # extend the index of the dataframe
# data = np.empty(len(index), dtype=dtypes)
# def df_empty(columns, dtypes, index=None):
    # assert len(columns)==len(dtypes)
    # df = pd.DataFrame(index=index)
    # for c,d in zip(columns, dtypes):
        # df[c] = pd.Series(dtype=d)
    # return df
# df = df_empty(columns, dtypes=[int, float, float, float, float])#, index=index)
# print(list(df.dtypes)) # int64, int64
# df = pd.DataFrame(data, index=index)


if INCEPTION_SCORE_FLAG:
    import tflib
    import tflib.inception_score
    def get_inception_score():
        all_samples = []
        samples = torch.randn(N_SAMPLES, N_LATENT)
        for i in range(0, N_SAMPLES, 100):
            samples_100 = samples[i:i+100].cuda(0)
            all_samples.append(gen(samples_100).cpu().data.numpy())

        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
        all_samples = all_samples.reshape((-1, N_CHANNEL, RESOLUTION, RESOLUTION)).transpose(0, 2, 3, 1)
        return tflib.inception_score.get_inception_score(list(all_samples))

    inception_f = open(os.path.join(OUTPUT_PATH, 'inception.csv'), 'ab')
    inception_writter = csv.writer(inception_f)


# if MODEL == "resnet":
    # gen = models.ResNet32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, BATCH_NORM_G)
    # dsc = models.ResNet32Discriminator(N_CHANNEL, 1, N_FILTERS_D, BATCH_NORM_D)
# elif MODEL == "dcgan":
    # gen = models.DCGAN32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, batchnorm=BATCH_NORM_G)
    # dsc = models.DCGAN32Discriminator(N_CHANNEL, 1, N_FILTERS_D, batchnorm=BATCH_NORM_D)
# elif MODEL == "custom":
gen = models.TwoLayerNet(N_LATENT, NHID)
# gen = DCGAN32Generator
# gen = models.DCGAN32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, batchnorm=BATCH_NORM_G)
# dsc = models.DCGAN32Discriminator(N_CHANNEL, 1, N_FILTERS_D, batchnorm=BATCH_NORM_D)
dsc = models.DeepCNNDiscriminator((N_CHANNEL, RESOLUTION, RESOLUTION), ln=LIPSCHITZ_NORMALIZER)

if "model" in checkpoint.keys():
    gen.load_state_dict(checkpoint["model"]["gen"])
    dsc.load_state_dict(checkpoint["model"]["dsc"])

elif "models" in checkpoint.keys():
    gen.load_state_dict(checkpoint["models"]["gen"])
    dsc.load_state_dict(checkpoint["models"]["dsc"])

gen.to(device)
dsc.to(device)


def get_checkpoint():
    '''Get current checkpoint'''
    global gen, dsc, stats, df, args, gen_optimizer, dsc_optimizer,  ep, n_gen_update, n_dsc_update, z_examples, total_time
    model = {'gen': gen.state_dict(), 'dsc': dsc.state_dict()}
    optimizer = {'gen': gen_optimizer.state_dict(), 'dsc': dsc_optimizer.state_dict()}
    niter = {'ep': ep, 'n_gen_update': n_gen_update, 'n_dsc_update': n_dsc_update}


    checkpoint = {
        'models': model,
        'df': df,
        'args' : args,
        'optimizers': optimizer,
        # 'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'niter': niter,
        'z_examples': z_examples.detach().cpu(),
        'total_time': total_time
                }

    return checkpoint

# def save_checkpoint(checkpoint=None, name=None, fname=None):
def save_checkpoint(checkpoint=None):
    '''Save checkpoint to disk'''

    global OUTPUT_PATH, ep, prev_n_gen, n_gen_update

    fname_prev_checkpoint = os.path.join(OUTPUT_PATH, "checkpoints", f"{prev_n_gen}.pth")
    if os.path.isfile(fname_prev_checkpoint):
        os.remove(fname_prev_checkpoint)

    name=f"checkpoints/{n_gen_update}"

    fname = os.path.join(OUTPUT_PATH, name + ".pth")

    # if fname is None:
        # fname = os.path.join(OUTPUT_PATH, name + '.pth')

    if checkpoint is None:
        checkpoint = get_checkpoint()

    torch.save(checkpoint, fname)
    prev_n_gen = n_gen_update

# gen.apply(lambda x: utils.weight_init(x, mode='normal'))
# dsc.apply(lambda x: utils.weight_init(x, mode='normal'))

dsc_optimizer = optim.SGD(dsc.parameters(), lr=LEARNING_RATE_D)
gen_optimizer = optim.SGD(gen.parameters(), lr=LEARNING_RATE_G)
# dsc_normalizer = LipschitzNormalizer(dsc, niter=LIPSCHITZ_NORMALIZER)
if "optimizer" in checkpoint.keys():
    gen_optimizer.load_state_dict(checkpoint["optimizer"]["gen"])
    dsc_optimizer.load_state_dict(checkpoint["optimizer"]["dsc"])
elif "optimizers" in checkpoint.keys():
    gen_optimizer.load_state_dict(checkpoint["optimizers"]["gen"])
    dsc_optimizer.load_state_dict(checkpoint["optimizers"]["dsc"])

if resf.tell() == 0:  # only print for new runs, with no results
    print("dsc", dsc, file=logf)
    print("gen", gen, file=logf)
    print("optimizer gen", gen_optimizer, file=logf)
    print("optimizer dsc", dsc_optimizer, file=logf, flush=True)


with open(os.path.join(OUTPUT_PATH, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

dataiter = iter(testloader)
examples, labels = next(dataiter)
torchvision.utils.save_image(utils.unormalize(examples), os.path.join(OUTPUT_PATH, 'examples.png'), nrow=10)

if "z_examples" in checkpoint.keys():
    z_examples = checkpoint["z_examples"].to(device)
else:
    z_examples = utils.sample(DISTRIBUTION, (100, N_LATENT)).to(device)


# gen_param_avg = []
# gen_param_ema = []
# for param in gen.parameters():
    # gen_param_avg.append(param.data.clone())
    # gen_param_ema.append(param.data.clone())

fieldnames = ("niter gen", "loss gen", "loss dsc", "GP", "time")
f_writer = csv.DictWriter(resf, fieldnames=fieldnames)
# f_writer = csv.DictWriter(resf)
if resf.tell() ==  0:
    f_writer.writeheader()
    #f_writer.writerow(fieldnames)

print('Training...')
n_iteration_t = 0
gen_inception_score = 0
while n_gen_update < N_ITER+start_n_gen:
    t = time.time()  # epoch time
    avg_loss_G = 0
    avg_loss_D = 0
    avg_penalty = 0
    d_samples = 0
    g_samples = 0
    penalty = torch.zeros(1, device=device)
    for i, data in enumerate(trainloader):
        _t = time.time()  # batch time
        x_true, _ = data
        x_true = x_true.to(device)

        z =utils.sample(DISTRIBUTION, (len(x_true), N_LATENT)).to(device)

        # if LIPSCHITZ_NORMALIZER:
            # dsc_normalizer.normalize()


        x_gen = gen(z)
        p_true, p_gen = dsc(x_true), dsc(x_gen)

        if UPDATE_FREQUENCY==1 or (n_iteration_t+1)%UPDATE_FREQUENCY != 0:
            # update dsc
            for p in gen.parameters():
                p.requires_grad = False


            dsc_optimizer.zero_grad()

            dsc_loss = - utils.compute_gan_loss(p_true, p_gen, mode=MODE)

            if GRADIENT_PENALTY:
                penalty = dsc.get_penalty_out(x_true.data, x_gen.data)

            loss = dsc_loss + 1/args.beta * penalty

            if UPDATE_FREQUENCY == 1:  # alternating update
                loss.backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

            dsc_optimizer.step()

            # if MODE =='wgan' and not (LIPSCHITZ_NORMALIZER or GRADIENT_PENALTY):
                # for p in dsc.parameters():
                    # p.data.clamp_(-CLIP, CLIP)

            n_dsc_update += 1

            avg_loss_D += dsc_loss.item()*len(x_true)
            avg_penalty += penalty.item()*len(x_true)

            d_samples += len(x_true)

            for p in gen.parameters():
                p.requires_grad = True

            if UPDATE_FREQUENCY != 1:  # will be updated at the generator turn
                total_time += time.time() - _t


        if UPDATE_FREQUENCY==1 or (n_iteration_t+1)%UPDATE_FREQUENCY == 0:
            # update gen

            for p in dsc.parameters():
                p.requires_grad = False

            gen_optimizer.zero_grad()

            loss = utils.compute_gan_loss(p_true, p_gen, mode=MODE, gen_flag=True)
            loss.backward()

            gen_optimizer.step()

            avg_loss_G += loss.item()*len(x_true)

            n_gen_update += 1
            # for j, param in enumerate(gen.parameters()):
                # gen_param_avg[j] = gen_param_avg[j]*n_gen_update/(n_gen_update+1.) + param.data.clone()/(n_gen_update+1.)
                # gen_param_ema[j] = gen_param_ema[j]*BETA_EMA+ param.data.clone()*(1-BETA_EMA)

            g_samples += len(x_true)

            for p in dsc.parameters():
                p.requires_grad = True

            total_time += time.time() - _t

            if n_gen_update%EVAL_FREQ == 1 or n_gen_update >= N_ITER:
                if INCEPTION_SCORE_FLAG:
                    gen_inception_score = get_inception_score()[0]

                    inception_writter.writerow((n_gen_update, gen_inception_score, total_time))
                    inception_f.flush()

                    if TENSORBOARD_FLAG:
                        writer.add_scalar('inception_score', gen_inception_score, n_gen_update)


                save_checkpoint()

                # torch.save({'args': vars(args), 'n_gen_update': n_gen_update, 'total_time': total_time, 'state_gen': gen.state_dict(), 'gen_param_avg': gen_param_avg, ma}, os.path.join(OUTPUT_PATH, "checkpoints/%i.state"%n_gen_update))


        n_iteration_t += 1

    ep += 1
    avg_loss_G /= g_samples
    avg_loss_D /= d_samples
    avg_penalty /= d_samples

    # print('Iter: %i, Loss Generator: %.4f, Loss Discriminator: %.4f, Penalty: %.2e, IS: %.2f, Time: %.4f'%(n_gen_update, avg_loss_G, avg_loss_D, avg_penalty, gen_inception_score, time.time() - t), file=logf, flush=True)

    stats = {'loss gen': avg_loss_G,
             'loss dsc': avg_loss_D,
             'GP': avg_penalty,
             'niter gen': n_gen_update,
             'time': time.time() - t}
    df.loc[ep, :] =stats
    f_writer.writerow(stats)
    # f_writer.writerow((resdct))
    resf.flush()

    with torch.no_grad():
        x_gen = gen(z_examples)
        x = utils.unormalize(x_gen)
        torchvision.utils.save_image(x.data, os.path.join(OUTPUT_PATH, 'gen/%i.png' % n_gen_update), nrow=10)

        fname = os.path.join(OUTPUT_PATH, "plots/loss.pdf")
        plot(fname, df)
        Vs = gen.compute_svdvals()  # one for each layer
        fname = os.path.join(OUTPUT_PATH, f"plots/svd_gen_{n_gen_update}.pdf")
        plot(fname, Vs, n_gen_update)
        Vs = dsc.compute_svdvals()  # one for each layer
        fname = os.path.join(OUTPUT_PATH, f"plots/svd_dsc_{n_gen_update}.pdf")
        plot(fname, Vs, n_gen_update)

    if TENSORBOARD_FLAG:
        writer.add_scalar('loss_G', avg_loss_G, n_gen_update)
        writer.add_scalar('loss_D', avg_loss_D, n_gen_update)
        writer.add_scalar('penalty', avg_penalty, n_gen_update)

        x = torchvision.utils.make_grid(x.data, 10)
        writer.add_image('gen', x.data, n_gen_update)
save_checkpoint()
resf.close()
print("end at ", datetime.datetime.now(), file=logf)
print("total time:", datetime.timedelta(seconds=total_time), file=logf)
logf.close()
