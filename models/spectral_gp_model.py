import gpytorch
import torch
import math
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.random_variables import GaussianRandomVariable
from gpytorch.kernels import GridInterpolationKernel
from spectral_gp_kernel import SpectralGPKernel

class SpectralLatentFunction(gpytorch.GridInducingPointModule):
	def __init__(self, frequencies, density, ninterp, x_bounds):
		super(SpectralLatentFunction, self).__init__(grid_size = ninterp, grid_bounds = [x_bounds])
		self.mean_module = ConstantMean(constant_bounds = [-1e-5,1e-5])
		self.covar_module = SpectralGPKernel(frequencies, density)

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return GaussianRandomVariable(mean_x, covar_x)

class SpectralGPModel(gpytorch.GPModel):
	def __init__(self, frequencies, density, log_noise, ninterp, x_bounds):
		super(SpectralGPModel, self).__init__(GaussianLikelihood(log_noise_bounds=(log_noise, log_noise)))
		self.latent_function = SpectralLatentFunction(frequencies, density, ninterp, x_bounds)

	def forward(self, x):
		return self.latent_function(x)
