from FeatureCloud.app.engine.app import AppState, app_state, Role
import os.path as op
import time
import bios
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from Autoencoder_classes import Autoencoder, ClusteringLayer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'
output_file = 'result.csv'
log_file ='run_logs.txt'

INITIAL_STATE = 'initial'
COMPUTE_STATE = 'compute'
AGGREGATE_STATE = 'aggregate'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'

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
        self.store('log_file', op.join(OUTPUT_DIR, log_file))
        self.store('output_file', op.join(OUTPUT_DIR, output_file))

        self.log('Done reading configuration file.')
        self.log(f'Max Iterations: {max_iterations}')
        self.log('Reading training data...')

        df_meta_data = pd.read_csv(f'{INPUT_DIR}/{input_metadata}', sep = input_sep)
        df = pd.read_csv(f'{INPUT_DIR}/{input_data}', sep = input_sep).apply(pd.to_numeric, errors='coerce')

        self.store('dataframe', df)
        self.store('dataframe_metadata', df_meta_data)

        self.log('Initializing pre-trained model...')
        proteomics_tensor = torch.tensor(df.values, dtype=torch.float32)

        input_dim = proteomics_tensor.shape[1]
        self.log(f'INITIAL input_dim: ${input_dim}')
        latent_dim = 10

        n_clusters = len(np.unique(df_meta_data["condition"].values))

        # Initialize the autoencoder
        autoencoder = Autoencoder(input_dim, latent_dim)
        clustering_layer = ClusteringLayer(n_clusters, latent_dim)

        # Initialize optimizers
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        optimizer_dec = torch.optim.Adam(
            list(autoencoder.parameters()) + list(clustering_layer.parameters()), lr=1e-3
        )
        
        # Initialize k-means clustering
        self.log('Initializing cluster centers...')

        with torch.no_grad():
            latent_representations = autoencoder.encoder(proteomics_tensor).numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        kmeans.fit(latent_representations)
        initial_cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        clustering_layer.cluster_centers.data = initial_cluster_centers

        # Store initialized components
        self.store('model', autoencoder)
        self.store('optimizer', optimizer)
        self.store('optimizer_dec', optimizer_dec)
        self.store('optimizer_state', optimizer.state_dict())
        self.store('clustering_layer', clustering_layer)
        self.store('data_tensor', proteomics_tensor)

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

        self.log("Loading global weights for model...")
        model = self.load('model')
        state_dict = {name: torch.tensor(param) for name, param in weights.items()}
        model.load_state_dict(state_dict)
        self.log('Updated model with received weights.')

        self.log('Loading data...')
        df = self.load('dataframe')
        df_meta_data = self.load('dataframe_metadata')


        self.log('Preparing model training...')
        X_tensor = self.load('data_tensor')

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
        self.store('optimizer_state', optimizer.state_dict())


        self.log('Scoring model...')
        with torch.no_grad():
            z, _ = model(X_tensor)
            q = clustering_layer(z)
            cluster_assignments = torch.argmax(q, dim=1).numpy()
        ari = adjusted_rand_score(y, cluster_assignments)
        self.log(f'Adjusted Rand Index (ARI): {ari}')

        updated_weights = {name: param.data.cpu().numpy() for name, param in model.state_dict().items()}

        self.send_data_to_coordinator(updated_weights)
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
            state_dict_list = self.gather_data()
            agg_weights = {}
            keys = []
            for state_dict in state_dict_list:
                    self.log(state_dict)
                    for key, value in state_dict.items():
                        agg_weights[key] = agg_weights.get(key, 0) + value 
                        keys.append(key)
            self.log(agg_weights)
            # Update global model with aggregated weights
            self.log('Updating global model...')
            global_model = self.load('model')
            self.log(global_model)
            for name, param in global_model.state_dict().items():
                param.data = torch.tensor(agg_weights[name].mean())

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
        self.log('Cluster data...')
        df = pd.read_csv(f'{INPUT_DIR}/allData.csv', sep = ';')
        model = self.load('model')
        output_file = self.load('output_file')
        target_column = self.load('target_column')
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/"state_dict')
        # TODO: get final clusters

        with open(self.load('log_file'), 'w') as handle:
            handle.write('iterations:\t'+str(self.iteration_counter)+'\n')
            handle.write('runtime:\t' + str(time.monotonic()-self.start_time)+'\n')
            self.out = {self.load('').FINISHED: True}

            return TERMINAL_STATE
        

