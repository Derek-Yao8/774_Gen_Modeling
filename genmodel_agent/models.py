"""Small, explicit VAE implementations with genuine observation likelihoods."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def nb_log_prob(x, mean, dispersion):
    """NB2: variance = mean + mean**2 / dispersion."""
    mean = mean.clamp_min(1e-8)
    dispersion = dispersion.clamp_min(1e-4)
    log_total = (mean + dispersion).log()
    return (torch.lgamma(x + dispersion) - torch.lgamma(dispersion) - torch.lgamma(x + 1)
            + dispersion * (dispersion.log() - log_total) + x * (mean.log() - log_total))


class BaseVAE(nn.Module):
    def condition(self, y, x):
        if self.n_classes:
            if y is None:
                raise ValueError("Conditional model requires labels")
            return torch.cat([x, F.one_hot(y, self.n_classes).to(x.dtype)], dim=1)
        return x

    def loss(self, x, y=None, noise=None):
        mu, logvar = self.encode(x, y)
        logvar = logvar.clamp(-12, 12)
        noise = torch.randn_like(mu) if noise is None else noise
        z = mu + (0.5 * logvar).exp() * noise
        reconstruction = self.reconstruction_nll(x, z, y)
        kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(1)
        return (reconstruction + kl).mean(), reconstruction.mean(), kl.mean()


class ImageVAE(BaseVAE):
    sigma = 0.1

    def __init__(self, shape, latent_dim, hidden_dim, n_classes=0):
        super().__init__()
        h, w = shape[:2]
        c = shape[2] if len(shape) == 3 else 1
        self.n_classes, self.latent_dim = n_classes, latent_dim
        self.spatial = (h // 4, w // 4)
        width = max(8, hidden_dim // 4)
        self.channels = width * 2
        self.convs = nn.Sequential(nn.Conv2d(c, width, 4, 2, 1), nn.ReLU(),
                                   nn.Conv2d(width, self.channels, 4, 2, 1), nn.ReLU())
        flat = self.channels * self.spatial[0] * self.spatial[1]
        self.encoder = nn.Sequential(nn.Linear(flat + n_classes, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.projection = nn.Sequential(nn.Linear(latent_dim + n_classes, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, flat), nn.ReLU())
        self.decoder = nn.Sequential(nn.ConvTranspose2d(self.channels, width, 4, 2, 1), nn.ReLU(),
                                     nn.ConvTranspose2d(width, c, 4, 2, 1), nn.Sigmoid())

    def encode(self, x, y=None):
        h = self.encoder(self.condition(y, self.convs(x).flatten(1)))
        return self.mu(h), self.logvar(h)

    def decode(self, z, y=None):
        h = self.projection(self.condition(y, z)).reshape(-1, self.channels, *self.spatial)
        return self.decoder(h)

    def reconstruction_nll(self, x, z, y):
        mean = self.decode(z, y)
        return (0.5 * ((x - mean) / self.sigma).square()
                + math.log(self.sigma) + 0.5 * math.log(2 * math.pi)).flatten(1).sum(1)


class CountVAE(BaseVAE):
    def __init__(self, n_features, latent_dim, hidden_dim, n_classes=0):
        super().__init__()
        self.n_classes, self.latent_dim = n_classes, latent_dim
        self.encoder = nn.Sequential(nn.Linear(n_features + n_classes, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + n_classes, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_features))
        self.raw_dispersion = nn.Parameter(torch.zeros(n_features))

    @property
    def dispersion(self):
        return F.softplus(self.raw_dispersion) + 1e-4

    def encode(self, x, y=None):
        h = self.encoder(self.condition(y, torch.log1p(x)))
        return self.mu(h), self.logvar(h)

    def decode(self, z, library_size, y=None):
        proportions = F.softmax(self.decoder(self.condition(y, z)), dim=1)
        return proportions * library_size.reshape(-1, 1)

    def reconstruction_nll(self, x, z, y):
        mean = self.decode(z, x.sum(1), y)
        return -nb_log_prob(x, mean, self.dispersion).sum(1)


def build_model(config, shape, n_classes):
    from .flows import FLOW_NAMES, ExpressionFlow
    if config["name"] in FLOW_NAMES:
        return ExpressionFlow(config)
    args = dict(latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"], n_classes=n_classes)
    if config["name"] == "image_gan":
        return ImageGAN(shape, **args)
    if config["name"] == "image_vae":
        return ImageVAE(shape, **args)
    if config["name"] == "count_vae":
        return CountVAE(shape[0], **args)
    raise ValueError(f"No trainer implemented for {config['name']}")


class ImageGAN(nn.Module):
    """Small convolutional GAN; implicit density, optional labels on both networks."""
    def __init__(self, shape, latent_dim, hidden_dim, n_classes=0):
        super().__init__()
        self.latent_dim, self.n_classes = latent_dim, n_classes
        h,w=shape[:2]
        c=shape[2] if len(shape)==3 else 1
        width=max(8,hidden_dim//4)
        flat=2*width*(h//4)*(w//4)
        self.generator=nn.Sequential(nn.Linear(latent_dim+n_classes,hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim,flat),nn.ReLU(),nn.Unflatten(1,(2*width,h//4,w//4)),
            nn.ConvTranspose2d(2*width,width,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(width,c,4,2,1),nn.Tanh())
        self.discriminator=nn.Sequential(nn.Conv2d(c+n_classes,width,4,2,1),nn.LeakyReLU(.2),
            nn.Conv2d(width,2*width,4,2,1),nn.LeakyReLU(.2),nn.Flatten(),
            nn.Linear(flat,hidden_dim),nn.LeakyReLU(.2),nn.Linear(hidden_dim,1))

    def labels(self,y):
        if y is None: raise ValueError('Conditional GAN requires labels')
        return F.one_hot(y,self.n_classes).float()

    def decode(self,z,y=None):
        if self.n_classes: z=torch.cat([z,self.labels(y).to(z.dtype)],1)
        return self.generator(z)

    def discriminate(self,x,y=None):
        if self.n_classes:
            labels=self.labels(y).to(x.dtype)[:,:,None,None].expand(-1,-1,*x.shape[2:])
            x=torch.cat([x,labels],1)
        return self.discriminator(x).flatten()

    def discriminator_loss(self,x,z,y=None):
        real=self.discriminate(x,y)
        fake=self.discriminate(self.decode(z,y).detach(),y)
        return .5*(F.binary_cross_entropy_with_logits(real,torch.ones_like(real))+
                   F.binary_cross_entropy_with_logits(fake,torch.zeros_like(fake)))

    def generator_loss(self,z,y=None):
        fake=self.discriminate(self.decode(z,y),y)
        return F.binary_cross_entropy_with_logits(fake,torch.ones_like(fake))

    def loss(self,x,y=None):
        # Device probe only; actual optimization uses separate alternating updates.
        z=torch.randn(len(x),self.latent_dim,device=x.device)
        d=self.discriminator_loss(x,z,y)
        g=self.generator_loss(z,y)
        return d+g,d,g
