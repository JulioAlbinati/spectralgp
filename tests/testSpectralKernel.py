import torch
import matplotlib.pyplot as plt
import math
from gpytorch.kernels import RBFKernel
from torch.autograd import Variable

import sys
sys.path.append('../kernels')
from spectral_gp_kernel import SpectralGPKernel

# input data
x = Variable(torch.linspace(-100,100,201)).unsqueeze(1)

# generating kernel values
kernel = RBFKernel(log_lengthscale_bounds = (math.log(20),math.log(20)))
k = kernel.forward(x, Variable(torch.Tensor([0])).unsqueeze(1))

# extracting approximate spectral density
width = math.pi / 500
omega = torch.linspace(width/2, math.pi-width/2, 500)
s = torch.zeros(500)
for ii in range(500):
	s[ii] = torch.dot(k.data.squeeze(), torch.cos(x.data.squeeze() * omega[ii]))

# reconstructing kernel
kernel_rec = SpectralGPKernel(omega.unsqueeze(1), s.unsqueeze(1))
k_rec = kernel_rec.forward(x, Variable(torch.Tensor([0])).unsqueeze(1))

# calculate difference
print torch.norm(k_rec - k)

plt.plot(x.data.numpy(), k_rec.data.numpy())
plt.plot(x.data.numpy(), k.data.numpy())
plt.show()
