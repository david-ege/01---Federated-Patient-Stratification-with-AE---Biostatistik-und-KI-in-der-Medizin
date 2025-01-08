from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import bios
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import scipy as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch import nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'
output_file = 'result.csv'

INITIAL_STATE = 'initial'
COMPUTE_STATE = 'compute'
AGGREGATE_STATE = 'aggregate'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
            ######### NEW ##############
            # nn.Sigmoid()
            ############################
        )

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

# Step 3: Target Distribution Update
def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()



@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE)  

    def run(self):
        self.log('Reading configuration file...')
        config = bios.read(f'{INPUT_DIR}/config.yml')

        self.log(f'Initializing: Setting iteration to 0, setting max_iterations...')
        self.store('iteration', 0)
        max_iterations = config['max_iter']
        input_data = config['data']
        input_metadata = config['metadata']
        input_sep = config['sep']
        target_column = config['target_value']
        self.store('max_iterations', max_iterations)
        self.store('target_column', target_column)
        self.store('input_file', input_data)
        self.store('input_metadata', input_metadata)
        self.store('input_sep', input_sep)
        self.log('Done reading configuration file.')
        self.log(f'Max Iterations: {max_iterations}')

        self.log('Reading training data...')

        df_meta_data = pd.read_csv(f'{INPUT_DIR}/{input_metadata}', sep= '\t')
        df_meta_data = df_meta_data.rename(columns={"sample_id": "file", "subclass": "condition", "class": "lab"})

        df = pd.read_csv(f'{INPUT_DIR}/{input_data}', sep = '\t', index_col=0)
        df = df.loc[:, df_meta_data['file']]

        self.store('dataframe', df)
        self.store('dataframe_metadata', df_meta_data)

        self.log('Initializing pre-trained model...')
        data_over_all_labs = (df.T - df.mean(axis=1))/df.std(axis=1)
        proteomics_tensor = torch.tensor(data_over_all_labs.values, dtype=torch.float32)

        X_tensor = proteomics_tensor
        proteomics_dl = DataLoader(X_tensor, batch_size=64, shuffle=True)
        input_dim = proteomics_tensor.shape[1]
        latent_dim = 10
        pre_training_epochs = 50
        train_epochs = 100

        y = df_meta_data["condition"].values
        n_clusters = len(np.unique(y))


        autoencoder = Autoencoder(input_dim, latent_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

        for epoch in range(pre_training_epochs):  # Pre-train for 50 epochs
            z, X_reconstructed = autoencoder(X_tensor)
            loss = criterion(X_reconstructed, X_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                self.log(f"Epoch {epoch+1}, Reconstruction Loss: {loss.item()}")

        self.log('Initializing clusterer...')
        # init the clusterer through kmeans
        with torch.no_grad():
            latent_representations = autoencoder.encoder(X_tensor).numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        kmeans.fit(latent_representations)
        initial_cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        clustering_layer = ClusteringLayer(n_clusters, latent_dim)
        clustering_layer.cluster_centers.data = initial_cluster_centers

        optimizer_dec = torch.optim.Adam(list(autoencoder.parameters()) + list(clustering_layer.parameters()), lr=1e-3)

        self.log('Training model')
        for epoch in range(train_epochs):  # Train DEC for 50 epochs
            z, X_reconstructed = autoencoder(X_tensor)
            q = clustering_layer(z)
            p = target_distribution(q)
            # KL Divergence Loss
            kl_loss = torch.nn.functional.kl_div(q.log(), p, reduction='batchmean')
            recon_loss = criterion(X_reconstructed, X_tensor)  # Optionally combine with reconstruction loss
            loss = kl_loss + 0.1 * recon_loss
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()
            if (epoch + 1) % 10 == 0:
                self.log(f"Epoch {epoch+1}, Total Loss: {loss.item()}, KL Loss: {kl_loss.item()}, Recon Loss: {recon_loss.item()}")

        self.log('Storing model, optimizer and latent space...')
        self.store('model', autoencoder)
        self.store('optimizer_state', optimizer.state_dict())
        self.store('optimizer_dec', optimizer_dec)
        self.store('clustering_layer', clustering_layer)

        if self.is_coordinator:
            self.log("Broadcasting initial weights to the clients...")
            weights = {name: param.data.cpu().numpy() for name, param in autoencoder.state_dict().items()}
            self.broadcast_data([weights, False])

        return COMPUTE_STATE


@app_state(COMPUTE_STATE)
class ComputeState(AppState):
    def register(self):
        self.register_transition(COMPUTE_STATE, role=Role.PARTICIPANT)
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)
        self.register_transition(WRITE_STATE)

    def run(self):
        iteration = self.load('iteration')
        iteration += 1
        self.store('iteration', iteration)

        self.log(f'ITERATION {iteration}')

        self.log('receiving weights...')
        weights, done = self.await_data()

        if done:
                return WRITE_STATE

        #TODO: Currently, we are getting a runtime error here: File "/root/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 2215, in load_state_dict
            #raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            #RuntimeError: Error(s) in loading state_dict for Autoencoder:
                #size mismatch for encoder.0.weight: copying a param with shape torch.Size([500, 2999]) from checkpoint, the shape in current model is torch.Size([500, 3000]).
                #size mismatch for decoder.6.weight: copying a param with shape torch.Size([2999, 500]) from checkpoint, the shape in current model is torch.Size([3000, 500]).
                #size mismatch for decoder.6.bias: copying a param with shape torch.Size([2999]) from checkpoint, the shape in current model is torch.Size([3000]).

        self.log("Loading global weights for model...")
        model = self.load('model')
        state_dict = {name: torch.tensor(param) for name, param in weights.items()}
        model.load_state_dict(state_dict)
        self.log('Updated model with received weights.')

        self.log('Loading data...')
        df = self.load('dataframe')
        df_meta_data = self.load('dataframe_metadata')


        self.log('Preparing model training...')
        data_over_all_labs = (df.T - df.mean(axis=1))/df.std(axis=1)
        proteomics_tensor = torch.tensor(data_over_all_labs.values, dtype=torch.float32)

        X_tensor = proteomics_tensor
        train_epochs = 100

        y = df_meta_data["condition"].values

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(self.load('optimizer_state'))

        self.log('Training model for one epoch...')
        for epoch in range(1):  
            z, X_reconstructed = model(X_tensor)
            loss = criterion(X_reconstructed, X_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.log(f'Local training loss: {loss.item()}')

        clustering_layer = self.load('clustering_layer')
        optimizer_dec = self.load('optimizer_dec')

        self.log('Training the decoder...')
        for epoch in range(train_epochs):  # Train DEC for 50 epochs
            z, X_reconstructed = model(X_tensor)
            q = clustering_layer(z)
            p = target_distribution(q)
            # KL Divergence Loss
            kl_loss = torch.nn.functional.kl_div(q.log(), p, reduction='batchmean')
            recon_loss = criterion(X_reconstructed, X_tensor)  # Optionally combine with reconstruction loss
            loss = kl_loss + 0.1 * recon_loss
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()
            if (epoch + 1) % 10 == 0:
                self.log(f"Epoch {epoch+1}, Total Loss: {loss.item()}, KL Loss: {kl_loss.item()}, Recon Loss: {recon_loss.item()}")

        self.log('Saving model and decoder...')
        self.store('model', model)
        self.store('optimizer_dec', optimizer_dec)

        self.log('Scoring model...')
        with torch.no_grad():
            z, _ = model(X_tensor)
            q = clustering_layer(z)
            cluster_assignments = torch.argmax(q, dim=1).numpy()
        ari = adjusted_rand_score(y, cluster_assignments)
        self.log(f'Adjusted Rand Index (ARI): {ari}')

        updated_weights = {name: param.data.cpu().numpy() for name, param in model.state_dict().items()}

        self.send_data_to_coordinator([updated_weights])
        self.log('Sent updated weights to coordinator.')

        if self.is_coordinator:
            return AGGREGATE_STATE
        else:
            return COMPUTE_STATE
        
@app_state(AGGREGATE_STATE)
class AggregateState(AppState):
    def register(self):
        self.register_transition(COMPUTE_STATE)

    def run(self):
            self.log('Waitig for local models...')
            agg_weight_lists = self.aggregate_data()
            agg_weights = {}

            for key in agg_weight_lists[0]:  # Iterate over parameter names
                agg_weight_lists[key] = sum(client_weights[key] for client_weights in agg_weight_lists) / len(agg_weight_lists)

            # Update global model with aggregated weights
            self.log('Updating global model...')
            global_model = self.load('model')
            for name, param in global_model.state_dict().items():
                param.data = torch.tensor(agg_weights[name])

            self.store('model', global_model)
            done = self.load('iteration') >= self.load('max_iterations')

            self.log('Broadcasting model...')
            self.broadcast_data([agg_weights, done])
            return COMPUTE_STATE
        
@app_state(WRITE_STATE)
class WriteState(AppState):
    def register(self):
        self.register_transition('terminal')

    def run(self):
            self.log('Predicting data...')
            #TODO: Predicting data and writing final results .csv
            #Old code:
            #df = self.load('dataframe')
            #model = self.load('model')
            #output_file = self.load('output_file')
            #target_column = self.load('target_column')
            #pred = model.predict(df.drop(columns = target_column))
            #pd.DataFrame(data={'pred': pred}).to_csv(f'{OUTPUT_DIR}/{output_file}')


            return TERMINAL_STATE
        

