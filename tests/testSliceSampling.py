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
from slice_sampling import SliceSampler

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

# defining hypers prior and initial value
h = torch.Tensor([0.27, 1.02, -6.43, -2.57, -0.05, 0.69, -2.07]).unsqueeze(1)
hmean = torch.Tensor([0.40, 1.19, -9.51, -2.52, -0.05, 0.69, -2.07]).unsqueeze(1)
hvars = torch.Tensor([0.20, 0.60, 5.00, 1.25, 0.02, 0.35, 1.00]).unsqueeze(1)
hinit = torch.zeros(7,1)

# defining frequencies
nfreq = 500
jitter = 1e-4
width = math.pi / nfreq
omega = torch.linspace(width/2, math.pi-width/2, nfreq).unsqueeze(1) + torch.randn(nfreq,1)*jitter

# reading approximate spectral density
s = []
fspectral = open('spectral.txt')
for line in fspectral:
	s.append(float(line.strip()))
s = torch.Tensor(s).unsqueeze(1)

# defining latent variables
mu = (h[1]*2).exp() * (-0.5*omega.pow(2)/(h[0]*2).exp()).exp() + h[2]
g = s.log() - mu + torch.randn(nfreq,1)*0.5

# defining likelihood function
def ell(h, g, omega, x, y, ninterp, x_bounds, hmean, hvars):
	# performs whitening and calculate p(nu) ~ N(0,I)
	nfreq = g.nelement()
	Kg = (h[4]*2).exp() * (-0.5*(omega - omega.t()).pow(2)/(h[3]*2).exp()).exp() + (h[5]*2).exp()*torch.eye(nfreq)
	Lg = torch.Tensor(cholesky(Kg.numpy()))
	nu = torch.mm(Lg.inverse(), g)
	pnu = -nfreq/2*math.log(2*math.pi) - 0.5*torch.mm(nu.t(), nu)
	print pnu
	# calculate p(h) ~ N(hmean, Diag(hvar))
	hdelta = h - hmean
	ph = -3.5*math.log(2*math.pi) - ((hvars.pow(0.5)).log()).sum() - 0.5*torch.mm(hdelta.t(), torch.mul(1/hvars, hdelta))
	print ph
	# calculate p(y)
	mu = (h[1]*2).exp() * (-0.5*omega.pow(2)/(h[0]*2).exp()).exp() + h[2]
	density = (g+mu).exp()
	model = SpectralGPModel(omega, density, h[6], ninterp, x_bounds)
	model.train()
	output = model(x)
	py = model.marginal_log_likelihood(output, y).data
	print py*y.nelement()
	return (pnu + ph + py*y.nelement())[0,0]
x_bounds = (1,500)
ell_func = lambda h : ell(h, g, omega, x, y, 500, x_bounds, hmean, hvars)

print ell_func(h)

# running slice sampling
#nsamples = 30
#width = hvars / 2
#ss = SliceSampler(hinit, width, ell_func, nsamples)
#ss.run()

#plt.plot(torch.linspace(1,nsamples,nsamples).numpy(), ss.ell.numpy())
#plt.show()
