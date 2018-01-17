import torch
import math
import warnings
from torch.autograd import Variable
from spectral_gp_model import SpectralGPModel

class EllipticalSliceSampler:
	def __init__(self, f_init, f_priors, model_ell_func, n_samples):
		self.n = f_init.nelement()
		self.f_init = torch.Tensor(f_init).unsqueeze(1)
		self.f_priors = torch.Tensor(f_priors)
		self.model_ell_func = model_ell_func
		self.n_samples = n_samples

	def run(self):
		self.f_sampled = torch.zeros(self.n, self.n_samples)
		self.ell = torch.zeros(self.n_samples, 1)

		f_cur = torch.Tensor(self.f_init)
		for ii in range(self.n_samples):
			print 'ess iteration %d' % ii
			if ii == 0:
				ell_cur = self.model_ell_func(f_cur)
			else:
				f_cur = torch.Tensor(self.f_sampled[:,ii-1]).unsqueeze(1)
				ell_cur = self.ell[ii-1, 0]
			next_f_prior = torch.Tensor(self.f_priors[:,ii]).unsqueeze(1)
			self.f_sampled[:,ii],self.ell[ii] = self.step(f_cur, next_f_prior, ell_cur)
			

	def step(self, f_cur, f_prior, ell_cur):
		theta = torch.rand(1) * 2*math.pi;
		theta_min = theta - 2*math.pi;
		theta_max = theta

		threshold = ell_cur + torch.rand(1).log()[0]/self.n 
		while True:
			f_new = f_cur * torch.cos(theta) + f_prior * torch.sin(theta)
			ell_new = self.model_ell_func(f_new)

			print 'threshold: %s ell: %s' % (threshold, ell_new)
			if ell_new > threshold:
				return (f_new, ell_new)
			else:
				if theta[0] < 0:
					theta_min = theta
				else:
					theta_max = theta
				interval = theta_max - theta_min
				if interval[0] < 1e-6:
					warnings.warn('ESS converged to initial point and sample was not accepted! (%s,%s)' % (ell_new, threshold))
					if math.isnan(ell_new):
						return (f_cur, ell_cur)
					else:
						return (f_cur, ell_new)
				theta = torch.rand(1) * interval + theta_min
