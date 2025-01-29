import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU(),
            nn.Linear(hidden_dim_3, latent_dim)
        )
    
    def forward(self, x):
        return self.fc(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_3),
            nn.ReLU(),
            nn.Linear(hidden_dim_3, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim)
        )
    
    def forward(self, z):
        return self.fc(z)

# Define GOAE Model
class GOAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3):
        super(GOAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3)
    
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, latent_dim):
        super(ClusteringLayer, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, z):
        # Compute soft assignments (q_ij)
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2))
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

def extract_latent_space(model, data_loader):
    model.encoder.fc.eval()
    latent_representations = []
    with torch.no_grad():
        for batch in data_loader:
            bach_latent_reepresentation = model.encoder(batch)
           
            latent_representations.append(bach_latent_reepresentation)
    return np.vstack(latent_representations)

# Orthogonality Loss
def orthogonality_loss(z):
    zt_z = torch.matmul(z.T, z)
    identity = torch.eye(zt_z.size(0), device=z.device)
    return torch.norm(zt_z - identity, p='fro')**2