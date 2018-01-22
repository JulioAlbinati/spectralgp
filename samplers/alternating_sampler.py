import torch
from numpy.linalg import cholesky

class AlternatingSampler:
	def __init__(self, outer_sampler_factory, inner_sampler_factory, nsamples_outer, nsamples_inner, cov_func, h_init, g_init):
		self.outer_sampler_factory = outer_sampler_factory
		self.inner_sampler_factory = inner_sampler_factory
		self.nsamples_outer = nsamples_outer
		self.nsamples_inner = nsamples_inner
		self.cov_func = cov_func
		self.h_init = torch.Tensor(h_init)
		if len(self.h_init.size()) < 2:
			self.h_init = self.h_init.unsqueeze(1)
		self.g_init = torch.Tensor(g_init)
		if len(self.g_init.size()) < 2:
			self.g_init = self.g_init.unsqueeze(1)
		self.ng = self.g_init.nelement()
		self.nh = self.h_init.nelement()

	def run(self):
		total_samples = (self.nsamples_inner+1) * self.nsamples_outer
		self.g_sampled = torch.zeros(self.ng, total_samples)
		self.h_sampled = torch.zeros(self.nh, total_samples)
		self.ell = torch.zeros(total_samples, 1)

		kk = 0
		for ii in range(self.nsamples_outer):
			print 'Obtaining samples %d to %d (out of a total of %d)...' % (kk, kk+self.nsamples_inner, total_samples)
			if ii == 0:
				g_cur = torch.Tensor(self.g_init)
				h_cur = torch.Tensor(self.h_init)
			else:
				g_cur = torch.Tensor(self.gsampled[:, kk-1]).unsqueeze(1)
				h_cur = torch.Tensor(self.hsampled[:, kk-1]).unsqueeze(1)
			# running update on hyperparameters
			nu_cur = self.whitening(g_cur, h_cur)
			outer_sampler = self.outer_sampler_factory(nu_cur, h_cur)
			outer_sampler.run()
			self.hsampled[:,kk].copy_(outer_sampler.hsampled[:, 0])
			self.gsampled[:,kk] = self.unwhitening(nu_cur, self.hsampled[:,kk].unsqueeze(1))
			self.ell[kk, 0] = outer_sampler.ell[0, 0]
			kk = kk + 1
			# running updates on latent variables
			g_cur = torch.Tensor(self.gsampled[:,kk-1]).unsqueeze(1)
			h_cur = torch.Tensor(self.hsampled[:,kk-1]).unsqueeze(1)
			inner_sampler = self.inner_sampler_factory(g_cur, h_cur, nsamples_inner)
			inner_sampler.run()
			self.gsampled[:, kk:(kk+nsamples_inner)].copy_(inner_sampler.gsampled)
			self.hsampled[:, kk:(kk+nsamples_inner)].copy_(inner_sampler.hsampled.expand(self.nh, nsamples_inner))
			self.ell[kk:(kk+nsamples_inner), 0] = outer_sampler.ell[:, 0]
			kk = kk + nsamples_inner

	def whitening(self, g, h):
		Kg = self.cov_func(h)
	        Lg = torch.Tensor(cholesky(Kg.numpy()))
        	return torch.mm(Lg.inverse(), g)

	def unwhitening(self, nu, h):
		Kg = self.cov_func(h)
	        Lg = torch.Tensor(cholesky(Kg.numpy()))
        	return torch.mm(Lg, nu)
