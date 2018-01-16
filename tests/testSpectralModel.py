import torch
import matplotlib.pyplot as plt
import math
import gpytorch
from numpy.linalg import cholesky
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.random_variables import GaussianRandomVariable
from torch.autograd import Variable

import sys
sys.path.append('../kernels')
from spectral_gp_kernel import SpectralGPKernel
sys.path.append('../models')
from spectral_gp_model import SpectralGPModel

# input data
x = Variable(torch.linspace(-500,500,1001)).unsqueeze(1)
y = Variable(torch.rand(1001,1))

# generating kernel values
kernel = RBFKernel(log_lengthscale_bounds = (math.log(20),math.log(20)))
k = kernel.forward(x, Variable(torch.Tensor([0])).unsqueeze(1))

# creating model
class ExactGPModel(gpytorch.GPModel):
    def __init__(self):
        super(ExactGPModel,self).__init__(GaussianLikelihood(log_noise_bounds=(-2, -2)))
        self.mean_module = ConstantMean(constant_bounds=(0, 0))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(math.log(20), math.log(20)))
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)

model = ExactGPModel()
model.train()
output = model(x)
print model.marginal_log_likelihood(output, y.squeeze())

# extracting approximate spectral density
width = math.pi / 2000
omega = torch.linspace(width/2, math.pi-width/2, 2000)
s = torch.zeros(2000)
for ii in range(2000):
	s[ii] = torch.dot(k.data.squeeze(), torch.cos(x.data.squeeze() * omega[ii]))

# creating spectral model
x_bounds = (-500, 500)
model = SpectralGPModel(omega.unsqueeze(1), s.unsqueeze(1), -2, 1001, x_bounds)
model.train()
output = model(x)
print model.marginal_log_likelihood(output, y.squeeze())
