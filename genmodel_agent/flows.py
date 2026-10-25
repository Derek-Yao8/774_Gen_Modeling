"""Standalone data-to-base flows; inverses are used for generation, not training."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FLOW_NAMES = ('planar_flow', 'radial_flow', 'realnvp_flow')


def log_dequantize(x, indices, seed):
    """Stable per-observation jitter; independent of batch/subset ordering."""
    x=np.asarray(x,dtype=np.float64)
    noise=np.stack([np.random.default_rng(np.random.SeedSequence([seed,int(i)])).uniform(1e-6,1.,x.shape[1]) for i in indices])
    return np.log(x+noise)


class Planar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w=nn.Parameter(torch.randn(dim)*.05)
        self.u=nn.Parameter(torch.randn(dim)*.05)
        self.b=nn.Parameter(torch.zeros(()))
    def parameters_constrained(self):
        # Unit direction avoids division by a nearly-zero squared norm.
        w=F.normalize(self.w,dim=0,eps=1e-12)
        dot=(w*self.u).sum()
        target=-.95+F.softplus(dot)
        u=self.u+(target-dot)*w
        return w,u,(w*u).sum()
    def forward(self,x):
        w,u,a=self.parameters_constrained()
        h=torch.tanh(x@w+self.b)
        return x+h[:,None]*u,torch.log1p(a*(1-h.square()))
    def inverse(self,y):
        w,u,a=self.parameters_constrained()
        rhs=y@w+self.b
        lo=rhs-a.abs()-1;hi=rhs+a.abs()+1
        # Solve t + a*tanh(t) = rhs. Derivative >= .05 ensures unique root.
        for _ in range(64):
            mid=(lo+hi)/2
            left=mid+a*torch.tanh(mid)<rhs
            lo=torch.where(left,mid,lo);hi=torch.where(left,hi,mid)
        x=y-torch.tanh((lo+hi)/2)[:,None]*u
        recovered,_=self(x)
        if not torch.allclose(recovered,y,atol=2e-5,rtol=2e-5):
            raise ValueError('Planar inverse residual exceeds tolerance')
        return x


class Radial(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.center=nn.Parameter(torch.randn(dim)*.05)
        self.raw_alpha=nn.Parameter(torch.zeros(()))
        self.raw_beta=nn.Parameter(torch.zeros(()))
    def constrained(self):
        alpha=F.softplus(self.raw_alpha)+1e-3
        beta=-.95*alpha+F.softplus(self.raw_beta)
        return alpha,beta
    def forward(self,x):
        alpha,beta=self.constrained()
        delta=x-self.center;r=torch.linalg.vector_norm(delta,dim=1)
        factor=1+beta/(alpha+r)
        radial=1+beta*alpha/(alpha+r).square()
        return self.center+factor[:,None]*delta,(x.shape[1]-1)*factor.log()+radial.log()
    def inverse(self,y):
        alpha,beta=self.constrained()
        delta=y-self.center;s=torch.linalg.vector_norm(delta,dim=1)
        b=alpha+beta-s
        root=torch.sqrt(b.square()+4*alpha*s)
        r=torch.where(b>=0,2*alpha*s/(root+b).clamp_min(1e-12),(-b+root)/2)
        return self.center+delta/(1+beta/(alpha+r))[:,None]


class Coupling(nn.Module):
    def __init__(self,dim,hidden,parity):
        super().__init__()
        self.register_buffer('mask',((torch.arange(dim)+parity)%2).float())
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,2*dim))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def terms(self,x):
        scale,shift=self.net(x*self.mask).chunk(2,1)
        return 2*torch.tanh(scale)*(1-self.mask),shift*(1-self.mask)
    def forward(self,x):
        s,t=self.terms(x)
        return x*torch.exp(s)+t,s.sum(1)
    def inverse(self,y):
        s,t=self.terms(y)
        return (y-t)*torch.exp(-s)


class ExpressionFlow(nn.Module):
    def __init__(self,config):
        super().__init__()
        dim=config['selected_features'];self.latent_dim=dim;self.n_classes=0
        self.shift=nn.Parameter(torch.zeros(dim));self.scale=nn.Parameter(torch.zeros(dim))
        name=config['name']
        self.layers=nn.ModuleList([Planar(dim) if name=='planar_flow' else Radial(dim) if name=='radial_flow'
            else Coupling(dim,config['hidden_dim'],i%2) for i in range(config['flow_layers'])])
    def forward(self,x):
        s=3*torch.tanh(self.scale)
        z=(x-self.shift)*torch.exp(s);ld=s.sum().expand(len(x))
        for layer in self.layers:
            z,increment=layer(z);ld=ld+increment
        return z,ld
    def log_prob(self,x):
        z,ld=self(x)
        return -.5*(z.square()+math.log(2*math.pi)).sum(1)+ld
    def loss(self,x,y=None,noise=None):
        nll=-self.log_prob(x).mean()
        return nll,nll,nll.new_zeros(())
    def decode(self,z,y=None):
        for layer in reversed(self.layers):z=layer.inverse(z)
        return z*torch.exp(-3*torch.tanh(self.scale))+self.shift
