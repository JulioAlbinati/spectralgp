import torch
import warnings
import math

class SliceSampler:
	def __init__(self, f_init, width, model_ell_func, n_samples):
		self.n = f_init.nelement()
		self.f_init = torch.Tensor(f_init)
		if len(self.f_init.size()) < 2:
			self.f_init = self.f_init.unsqueeze(1)
		self.width = width
		self.model_ell_func = model_ell_func
		self.n_samples = n_samples

	def run(self):
		self.f_sampled = torch.zeros(self.n, self.n_samples)
		self.ell = torch.zeros(self.n_samples, 1)

		f_cur = torch.Tensor(self.f_init)
		for ii in range(self.n_samples):
			print 'iteration %d' % ii
			if ii == 0:
				ell_cur = self.model_ell_func(f_cur)
			else:
				f_cur = torch.Tensor(self.f_sampled[:,ii-1]).unsqueeze(1)
				ell_cur = self.ell[ii-1,0]

			self.f_sampled[:,ii], self.ell[ii,0] = self.step(f_cur, ell_cur)
			print 'ell = %s' % self.ell[ii,0]

	def step(self, f_cur, ell_cur):
		f_new = torch.Tensor(f_cur)
		ell_new = ell_cur
		for ii in range(self.n):
			print 'updating index %d' % ii
			threshold = ell_new + torch.rand(1).log()[0]

			rr = torch.rand(1)[0]
			f_max = f_new[ii,0] + (1-rr)*self.width[ii,0]
			f_min = f_new[ii,0] - rr*self.width[ii,0]
			f_cand = torch.Tensor(f_new)

			while True:
				f_cand[ii,0] = torch.rand(1)[0]*(f_max-f_min) + f_min
				ell_cand = self.model_ell_func(f_cand)

				if ell_cand > threshold:
					f_new[ii,0] = f_cand[ii,0]
					ell_new = ell_cand
					break
				elif f_cand[ii,0] > f_new[ii,0]:
					f_max = f_cand[ii,0]
				else:
					f_min = f_cand[ii,0]

				if (f_max-f_min) < 1e-6:
					warnings.warn('Slice sampling converged to initial point and sample was not accepted! (%s,%s)' % (ell_cand, threshold))
					if not math.isnan(ell_cand):
						ell_new = ell_cand
					break

		return (f_new, ell_new)
