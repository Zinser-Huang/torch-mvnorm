
#%%
import torch
from torch.autograd import grad
from mvnorm import multivariate_normal_cdf as Phi
#%%
import torch
from torch.autograd import grad
from mvnorm import multivariate_normal_cdf as Phi
n = 2
x = 1 + torch.randn(n)
x.requires_grad = True
# Make a positive semi-definite matrix
A = torch.randn(n,n)
C = 1/n*torch.matmul(A,A.t())
p = Phi(x,loc = torch.zeros(n), covariance_matrix=C)
print(p)

#%%

from scipy import stats

mvn = stats.multivariate_normal(mean=[0,0], cov=C.detach().numpy())
print(mvn.cdf(x.detach().numpy()))