import torch
from gpytorch.kernels.kernel import Kernel

class SpectralGPKernel(Kernel):
	def __init__(self, frequencies, density):
		super(SpectralGPKernel, self).__init__()
		s_freq = frequencies.size()[0]
		s_dens = density.size()[0]
		frequencies = frequencies.reshape(s_freq, 1)
		density = density.reshape(s_dens, 1)

		if frequencies.shape != density.shape:
			raise RuntimeError('Dimension mismatch of frequencies and density vector!')

		self.frequencies = frequencies
		self.density = density
		self.k = None

	def compute_kernel_values(self, n):
		s,_ = self.frequencies.size()
		Tau = torch.Tensor([range(0,n+1)]).t().expand(n+1,s)
		Omega = self.frequencies.t().expand(n+1,s)
		k = 1/float(s) * torch.mm(torch.cos(torch.mul(Omega, Tau)), self.density)
		return k

	def forward(self, x1, x2):
		if len(list(x1.shape)) == 1:
			x1 = x1.unsqueeze(1)
		if len(list(x2.shape)) == 1:
			x2 = x2.unsqueeze(1)

		n, d1 = x1.size()
		m, d2 = x2.size()

		if d1 > 1 or d2 > 1:
			raise RuntimeError(' '.join([
				'The spectral GP kernel can only be applied'
				'to a single dimension at a time. To use'
				'multi-dimensional data, use a product of SGP'
				'kernels, one for each dimension.'
			]))

		Tau = (x1 - x2.t()).abs()
		max_tau = int(Tau.max().item())
		if self.k is None or (self.k.size()[0]-1) < max_tau:	# extending kernel values if needed
			self.k = self.compute_kernel_values(max_tau)

		Tau.resize_(n*m,1)
		K = self.k[Tau.int().data.numpy().transpose().tolist()].resize(n,m)

		return K
