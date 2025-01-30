from FeatureCloud.app.engine.app import AppState, app_state, Role
import os.path as op
import time
import bios
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from Autoencoder_classes import Encoder, Decoder, GOAE, ClusteringLayer, extract_latent_space, orthogonality_loss
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
        epochs_per_iteration = config['epochs']

        self.store('epochs', epochs_per_iteration)
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
        df = pd.read_csv(f'{INPUT_DIR}/{input_data}', sep = input_sep, index_col=0)

        self.store('dataframe', df)
        self.store('dataframe_metadata', df_meta_data)

        self.log('Initializing pre-trained model...')


        sample_data_normalized = df.reset_index(drop=True).to_numpy(dtype=np.float32)
        sample_data_normalized = np.nan_to_num(sample_data_normalized)
        proteomics_tensor = torch.tensor(sample_data_normalized, dtype=torch.float32)

        input_dim = proteomics_tensor.shape[1]
        self.log(f'INITIAL input_dim: ${input_dim}')
        latent_dim = 10
        hidden_dim_1 = 500
        hidden_dim_2 = 2000
        hidden_dim_3 = 500
        n_clusters = 2

        # Initialize the autoencoder
        autoencoder = GOAE(input_dim, latent_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, hidden_dim_3=hidden_dim_3)
        #clustering_layer = ClusteringLayer(n_clusters, latent_dim)

        # Initialize k-means clustering
        self.log('Initializing cluster centers...')


        # Initialize optimizers
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

        # Store initialized components
        self.store('model', autoencoder)
        self.store('optimizer', optimizer)
        self.store('optimizer_state', optimizer.state_dict())
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
        optimizer = self.load('optimizer')
        state_dict = {name: torch.tensor(param) for name, param in weights.items()}
        model.load_state_dict(state_dict)
        self.log('Updated model with received weights.')

        self.log('Loading data...')
        df = self.load('dataframe')
        df_meta_data = self.load('dataframe_metadata')


        self.log('Preparing model training...')
        X_tensor = self.load('data_tensor')

        train_epochs = self.load('epochs')

        y = df_meta_data["Conditions"].values

        criterion = nn.MSELoss()
        
        optimizer.load_state_dict(self.load('optimizer_state'))

        proteomics_ds = TensorDataset(X_tensor)
        data_loader = DataLoader(proteomics_ds, batch_size=64, shuffle=True)
        lambda_ortho = 0.1
        self.log('Training model...')
        for epoch in range(train_epochs):  # Train DEC for 50 epochs

            total_loss = 0
            for x_batch in data_loader:
                x_batch = x_batch[0]
            
                # Forward pass
                z, x_reconstructed = model(x_batch)
                
                # Loss
                reconstruction_loss = criterion(x_reconstructed, x_batch)
                ortho_loss = orthogonality_loss(z)
                loss = reconstruction_loss + lambda_ortho * ortho_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                    self.log(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")


        clustering_layer = extract_latent_space(model, X_tensor)
        kmeans = KMeans(n_clusters=2, random_state=22)
        clusters = kmeans.fit_predict(clustering_layer)

        self.log('Saving model and decoder...')
        self.store('model', model)
        self.store('optimizer_state', optimizer.state_dict())
        self.store('clustering_layer', clustering_layer)
        self.store('clusters', clusters)

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
                    #self.log(state_dict)
                    for key, value in state_dict.items():
                        agg_weights[key] = agg_weights.get(key, 0) + ( value / len(state_dict_list))
                        keys.append(key)
            # Update global model with aggregated weights
            self.log('Updating global model...')
            global_model = self.load('model')
            #for name, param in global_model.state_dict().items():
            #    param.data = torch.tensor(agg_weights[name].mean())

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
        
        self.log('Saving final model...')
        model = self.load('model')
        output_file = self.load('output_file')
        target_column = self.load('target_column')
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/"state_dict')
        # TODO: get final clusters

        #with open(self.load('log_file'), 'w') as handle:
            #handle.write('iterations:\t'+str(self.iteration_counter)+'\n')
            #handle.write('runtime:\t' + str(time.monotonic()-self.start_time)+'\n')
            #self.out = {self.load('').FINISHED: True}

        return TERMINAL_STATE