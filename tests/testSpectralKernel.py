import torch
import matplotlib.pyplot as plt
import math
from gpytorch.kernels import RBFKernel

import sys
sys.path.append('../kernels')
from spectral_gp_kernel import SpectralGPKernel

# input data
x = torch.linspace(-100,100,201).unsqueeze(1)

# generating kernel values
kernel = RBFKernel()
kernel.lengthscale = 10.0
k = kernel.forward(x, torch.Tensor([0])).squeeze()

# extracting approximate spectral density
width = math.pi / 500
omega = torch.linspace(width/2, math.pi-width/2, 500)
s = torch.zeros(500)
for ii in range(500):
	s[ii] = torch.dot(k, torch.cos(x.data.squeeze() * omega[ii]))

# reconstructing kernel
kernel_rec = SpectralGPKernel(omega, s)
k_rec = kernel_rec.forward(x, torch.Tensor([0]))

# calculate difference
print(torch.norm(k_rec - k.unsqueeze(1)))

plt.plot(x.data.numpy(), k_rec.data.numpy())
plt.plot(x.data.numpy(), k.data.numpy())
plt.show()