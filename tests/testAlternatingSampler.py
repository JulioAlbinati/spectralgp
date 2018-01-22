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
from slice_sampling import SliceSampler
from alternating_sampler import AlternatingSampler

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
x_bounds = (1,500)
ninterp = 500

# reading hypers
h = torch.Tensor([0.27, 1.02, -6.43, -2.57, -0.05, 0.69, -2.07]).unsqueeze(1)
hmean = torch.Tensor([0.40, 1.19, -9.51, -2.52, -0.05, 0.69, -2.07]).unsqueeze(1)
hvars = torch.Tensor([0.20, 0.60, 5.00, 1.25, 0.02, 0.35, 1.00]).unsqueeze(1)

# defining frequencies
nfreq = 500
jitter = 1e-2
width = math.pi / nfreq
omega = torch.linspace(width/2, math.pi-width/2, nfreq).unsqueeze(1) + torch.randn(nfreq,1)*jitter

# sampling from prior
mu = (h[1]*2).exp() * (-0.5*omega.pow(2)/(h[0]*2).exp()).exp() + h[2]
Kg = (h[4]*2).exp() * (-0.5*(omega - omega.t()).pow(2)/(h[3]*2).exp()).exp() + (h[5]*2).exp()*torch.eye(nfreq)
Lg = torch.Tensor(cholesky(Kg.numpy()))
g = torch.mm(Lg, torch.randn(nfreq, 1))

# defining ESS factory
def ess_factory(g, h, nsamples, omega, x, y, ninterp, x_bounds):
	# calculating mean value
	mu = (h[1]*2).exp() * (-0.5*omega.pow(2)/(h[0]*2).exp()).exp() + h[2]
	# sampling from current prior
	Kg = (h[4]*2).exp() * (-0.5*(omega - omega.t()).pow(2)/(h[3]*2).exp()).exp() + (h[5]*2).exp()*torch.eye(nfreq)
	Lg = torch.Tensor(cholesky(Kg.numpy()))
	Gprior = torch.mm(Lg, torch.randn(nfreq, nsamples))
	# defining log-likelihood function
	def ell(g, omega, x, y, mu, log_noise, ninterp, x_bounds):
		density = (g+mu).exp()
		model = SpectralGPModel(omega, density, log_noise, ninterp, x_bounds)
		model.train()
		output = model(x)
		return model.marginal_log_likelihood(output, y).data[0]
	ell_func = lambda g : ell(g, omega, x, y, mu, h[6], ninterp, x_bounds)
	# creating model
	return EllipticalSliceSampler(g, Gprior, ell_func, nsamples)
ess_fact = lambda g,h,nsamples : ess_factory(g, h, nsamples, omega, x, y, ninterp, x_bounds)

# defining slice sampler factory
def ss_factory(nu, h, nsamples, omega, x, y, ninterp, x_bounds, hmean, hvars):
	# defining log-likelihood function
	def ell(h, nu, omega, x, y, ninterp, x_bounds, hmean, hvars):
		# performs whitening and calculate p(nu) ~ N(0,I)
		pnu = -nfreq/2*math.log(2*math.pi) - 0.5*torch.mm(nu.t(), nu)
		# calculate p(h) ~ N(hmean, Diag(hvar))
		hdelta = h - hmean
		ph = -3.5*math.log(2*math.pi) - ((hvars.pow(0.5)).log()).sum() - 0.5*torch.mm(hdelta.t(), torch.mul(1/hvars, hdelta))
		# calculate p(y)
		mu = (h[1]*2).exp() * (-0.5*omega.pow(2)/(h[0]*2).exp()).exp() + h[2]
		density = (g+mu).exp()
		model = SpectralGPModel(omega, density, h[6], ninterp, x_bounds)
		model.train()
		output = model(x)
		py = model.marginal_log_likelihood(output, y).data * y.nelement()
		return (pnu + ph + py)[0,0]
	ell_func = lambda h : ell(h, nu, omega, x, y, ninterp, x_bounds, hmean, hvars)
	# creating model
	return SliceSampler(h, hvars/2, ell_func, nsamples)
ss_fact = lambda g,nu : ss_factory(g, nu, 1, omega, x, y, ninterp, x_bounds, hmean, hvars)

# defining covariance function for latent variables
cov_func = lambda h : (h[4]*2).exp() * (-0.5*(omega - omega.t()).pow(2)/(h[3]*2).exp()).exp() + (h[5]*2).exp()*torch.eye(nfreq)

# running alternating sampler
nsamples_outer = 1000
nsamples_inner = 5
alt_sampler = AlternatingSampler(ss_fact, ess_fact, nsamples_outer, nsamples_inner, cov_func, h, g)
alt_sampler.run()

total_samples = (nsamples_inner+1) * nsamples_outer
plt.plot(torch.linspace(1,total_samples,total_samples).numpy(), alt_sampler.ell.numpy())
plt.show()
