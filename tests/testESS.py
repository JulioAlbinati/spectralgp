import torch
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable
from numpy.linalg import cholesky

import sys
sys.path.append('../models')
sys.path.append('../kernels')
sys.path.append('../samplers')
from spectral_gp_model import SpectralGPModel
from elliptical_slice_sampling import EllipticalSliceSampler

# reading training data
fdata = open('/media/jalbinati/DADOS/cornell/kernel-learning/synthetic-data/synthetic_1D_rbftimesper_500.txt')
x = []
y = []
for line in fdata:
	tokens = line.strip().split()
	x.append(float(tokens[0]))
	y.append(float(tokens[1]))
x = Variable(torch.Tensor(x))
y = Variable(torch.Tensor(y))

# reading hypers
h = torch.Tensor([0.27, 1.02, -6.43, -2.57, -0.05, 0.69, -2.07]).unsqueeze(1)

# defining frequencies
nfreq = 500
jitter = 1e-4
width = math.pi / nfreq
omega = torch.linspace(width/2, math.pi-width/2, nfreq).unsqueeze(1) + torch.randn(nfreq,1)*jitter

# sampling from prior
nsamples = 100
mu = (h[1]*2).exp() * (-0.5*omega.pow(2)/(h[0]*2).exp()).exp() + h[2]
Kg = (h[4]*2).exp() * (-0.5*(omega - omega.t()).pow(2)/(h[3]*2).exp()).exp() + (h[5]*2).exp()*torch.eye(nfreq)
Lg = torch.Tensor(cholesky(Kg.numpy()))
Gprior = torch.mm(Lg, torch.randn(nfreq, 100+1))

# defining likelihood function
def ell(g, omega, x, y, mu, log_noise, ninterp, x_bounds):
	density = (g+mu).exp()
	model = SpectralGPModel(omega, density, log_noise, ninterp, x_bounds)
	model.train()
	output = model(x)
	return model.marginal_log_likelihood(output, y).data[0]
x_bounds = (1,500)
ell_func = lambda g : ell(g, omega, x, y, mu, h[6], 500, x_bounds)

# running ESS
ess = EllipticalSliceSampler(Gprior[:,0], Gprior[:,1:], ell_func, nsamples)
ess.run()

plt.plot(torch.linspace(1,nsamples,nsamples).numpy(), ess.ell.numpy())
plt.show()
