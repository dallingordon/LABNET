import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import uniform, normal
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import random
import torch.nn.functional as F
import torch.distributions as dist


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

slm_init_config = {"embedding_dim" : 16
                    ,"num_heads" : 8
                    ,"hidden_dim"  : 11
                    ,"num_layers" : 2
                    ,"dropout" : 0.1
                    ,"vocab_size" : 80
                    ,"class_num" : 80
                    ,"sequence_length" : 160}

slm_model_config = {"dist_type" : "ints" ##lower was worse.  raise it. 0.003 looks great.  this is the best.
                      , "gen_m" : 80 #should be class_num, vocab_size
                      , "gen_n" : 2000
                      , "gen_epochs" : 50
                      , "gen_lr" :  0.003 ##0.003
                      , "random_shuffle" : 0.8
                      , "out_type" : "one-hot" }

def generate_heteroskedastic_ints(shape, alpha, beta, total_classes):
    # Generate random values from the beta distribution
    values = np.random.beta(alpha, beta, shape)
    
    # Scale the values to integers in the desired range (e.g., 0 to 100)
    min_value = 0
    max_value = total_classes
    scaled_values = min_value + (max_value - min_value) * values
    ints = np.round(scaled_values).astype(int)
    
    return ints

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )

    def forward(self, x):
        return self.transformer(x)

class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, dropout,sequence_length):
        super(ToyTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout, batch_first=True),
            num_layers
        )
        self.fc1 = nn.Linear(embedding_dim * sequence_length, vocab_size)  # Intermediate linear layer
        self.fc2 = nn.Linear(vocab_size, vocab_size)  # Final linear layer

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = torch.flatten(x, start_dim=1)  # Flatten the output
        x = F.relu(self.fc1(x))  # Apply the intermediate linear layer with ReLU activation
        x = self.fc2(x)  # Apply the final linear layer
        x = F.softmax(x, dim=1)  # Apply softmax activation
        return x

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, sequence_length, embedding_dim, class_num, num_heads, hidden_dim, num_layers, dropout):
        super(SimpleLanguageModel, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Define the transformer encoder
        self.transformer_encoder = TransformerEncoder(embedding_dim, num_heads, hidden_dim, num_layers, dropout)
        
        self.fc1 = nn.Linear(sequence_length*embedding_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        
        self.output_layer = nn.Linear(500, class_num)
        
        

    def forward(self, input_data):
        # Input_data is of shape (batch_size, sequence_length)
        # Apply embedding layer
        #print(input_data.shape)
        embedded = self.embedding(input_data)
        #print(embedded.shape)
        # Pass through the transformer encoder
        transformed = self.transformer_encoder(embedded)
        #print(transformed.shape) same as input duh: batch x sequence_length x embedding_dim
        flattened_tensor = transformed.view(-1,self.sequence_length*self.embedding_dim)
        f1 = nn.ReLU()(self.fc1(flattened_tensor))
        f2 = nn.ReLU()(self.fc2(f1))
        out = self.output_layer(f2)
        # Apply the output layer
        output = F.softmax(out,dim=1)

        return output
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

##This could maybe be a method of Lab?
def compare_rows(tensor_a, tensor_b, dist='euclidean'):
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("Input tensors must have the same shape.")
    if dist == 'euclidean':
        distances = euclidean_distances(tensor_a, tensor_b)
    elif dist == 'cosine':
        distances = cosine_distances(tensor_a, tensor_b)
    else:
        raise ValueError("Invalid distance metric. Please choose either 'euclidean' or 'cosine'.")

    min_sum = np.sum(np.min(distances, axis=1))
    return distances,min_sum

def one_hot_last_dim(tensor_shape):
    num_classes = tensor_shape[-1]
    random_idx = np.random.randint(1, num_classes + 1, size=tensor_shape[:-1])
    zero_tensor = np.zeros(tensor_shape, dtype=int)
    last_dim_indices = np.arange(num_classes)
    zero_tensor[..., :, last_dim_indices] = (random_idx[..., np.newaxis] == last_dim_indices)
    return zero_tensor

class Teacher:
    
    def __init__(self,layer_sizes,input_shape = None):
            
            
        self.cofigured = False
        self.input_shape = None
        self.output_shape = None
        self.train_inputs = torch.tensor([])
        self.train_targets = torch.tensor([])
        self.val_inputs = torch.tensor([])
        self.val_targets = torch.tensor([])
        self.list_config = None #true uses list config, not model config
        
        if isinstance(layer_sizes,nn.Module):
            if input_shape is None:
                raise ValueError('when initializing with a model include the input_shape as the second parameter')
            else:
                if isinstance(input_shape,tuple):
                    self.input_shape = input_shape
                else: 
                    raise ValueError('input_shape must be a tuple')
            self.model = layer_sizes
            
        
        if isinstance(layer_sizes, list):
            self.list_config = True
            num_inputs = layer_sizes[0]
            num_outputs = layer_sizes[-1]
            self.model = nn.Sequential()
            

            # Add input layer
            self.model.add_module("input_layer", nn.Linear(num_inputs, layer_sizes[1]))
            self.model.add_module("input_layer_activation", nn.ReLU())

            # Add hidden layers
            for i in range(2, len(layer_sizes)-1):
                self.model.add_module(f"hidden_layer_{i}", nn.Linear(layer_sizes[i-1], layer_sizes[i]))
                self.model.add_module(f"hidden_layer_{i}_activation", nn.ReLU())

            # Add output layer
            self.model.add_module("output_layer", nn.Linear(layer_sizes[-2], num_outputs))


    def load_state_dict(self,path):
        self.model.load_state_dict(torch.load(path))
        dummy_shape = (1,) + self.input_shape
        dummy_in = torch.ones(dummy_shape)
        try:
            dummy_out = self.model(dummy_in)
        except Exception as e:
            print(e)
            print("lets try ints!")
            dummy_in = torch.randint(low=0, high=1, size=dummy_shape)
            dummy_out = self.model(dummy_in)

        self.output_shape = tuple(dummy_out.shape[1:]) # don't need the one batch, tack it on out of the if
                                  
        self.cofigured = True
        self.model.eval()  
        
    def configure(self
                  , gen_lr = 0.01
                  , gen_epochs = 1000
                  , gen_init_range = (-1,1)
                  , gen_n = 10_000
                  , gen_m =0.0
                  , gen_std=1.0
                  , out_type = None
                  , batch_size = 50
                  , random_shuffle = 0.5
                  , dist_type = 'normal'
                  , alpha = 1
                  , beta = 1
                 ):
        
        low,high = gen_init_range
        
        #initialize the teacher weights, does all uniform rn.  
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                init.uniform_(module.weight, low, high)
                if module.bias is not None:
                    init.uniform_(module.bias, low, high)
        
        
        
        if self.list_config:
            self.input_shape = tuple([self.model.input_layer.in_features]) ##this only works with create_neural_network func above
            self.output_shape = tuple([self.model.output_layer.out_features])
        else:
            #this is to get shapes for model init
            dummy_shape = (1,) + self.input_shape
            dummy_in = torch.ones(dummy_shape)
            try:
                dummy_out = self.model(dummy_in)
            except Exception as e:
                print(e)
                print("lets try ints!")
                dummy_in = torch.randint(low=0, high=gen_m, size=dummy_shape)
                dummy_out = self.model(dummy_in)
            
            self.output_shape = tuple(dummy_out.shape[1:]) # don't need the one batch, tack it on out of the if
                                   
        gen_shape = (gen_n,) + self.input_shape  
        gen_out_shape = (gen_n,) + self.output_shape
        
        if out_type == "one-hot":
            out_temp = one_hot_last_dim(gen_out_shape)
        else:
            out_temp = np.random.normal(gen_m, gen_std, gen_out_shape)
            
        out_temp = torch.from_numpy(out_temp).float()
        #self.out_temp = out_temp #this was just to debug, see the training config outputs
        
            
            ##i need an output size as well! dg start here
            
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=gen_lr)
        
       
        #make temporary input data for getting good teacher weights.
        if dist_type == 'normal':
            samples = np.random.normal(gen_m, gen_std, gen_shape)
            samples = torch.from_numpy(samples).float()
        elif dist_type == 'uniform':
            samples = np.random.uniform(gen_m, gen_std, gen_shape)
            samples = torch.from_numpy(samples).float()
        elif dist_type == "ints":
            samples = np.random.randint(0, high=gen_m, size=gen_shape)
            samples = torch.from_numpy(samples)
        elif dist_type == "hetero":
            samples = generate_heteroskedastic_ints(gen_shape, alpha, beta,gen_m-1)
            samples = torch.from_numpy(samples)
        else:
            raise ValueError('dist_type must be normal, uniform, hetero, or ints.')
        
        

        dataset = TensorDataset(samples, out_temp)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        progress_bar = tqdm(total=gen_epochs, desc="Configuring Teacher:")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU is not available, using CPU")
        self.model.to(device)
        self.model.train()
        for epoch in range(gen_epochs):
            for batch_samples, batch_out_temp in dataloader:
                batch_samples, batch_out_temp = batch_samples.to(device), batch_out_temp.to(device)
                #this is an attempt to have more balanced outputs from my fake model
                random_number = random.random()
                if random_number < random_shuffle: # this means higher r_s is more shuffling.  makes more sense imo
                    batch_out_temp = np.take(batch_out_temp, np.random.permutation(batch_out_temp.shape[0]), axis=0)
                
                
                outputs = self.model(batch_samples)
                loss = criterion(outputs, batch_out_temp)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            progress_bar.update(1)

            #####add a progress bar! https://chat.openai.com/c/385d20e0-ebcd-4894-a356-7c6fd5c80913
        #print("Teacher Configured, now you can generate data!")
        self.cofigured = True
        progress_bar.close()
        self.model.eval()
        print("Teacher Configured")
    #this would be theoretical perfect dark knowledge
    def generate_data(self
                      , val_train = "train"
                      , n = 1000
                      , dist_type = 'normal'
                      , m =0.0
                      , batch_size = 50
                      , std=1.0
                      , display_progress = True
                      , alpha = 1
                      , beta = 1
                      , store_outputs = False
                     ):
        if val_train not in ["train","val"]:
            raise RuntimeError("please specify val_train = 'train' or 'val'.")

        if not self.cofigured:
            #if it is configured, we have self.input_shape and self.output shape
            raise RuntimeError("Teacher is not configured. Run the configure() method of your teacher object before generating.")
        
        input_size = self.input_shape ##this only works with create_neural_network func above
        #output_size = self.output_shape un used
        gen_shape = (n,) + input_size
        
        if dist_type == 'normal':
            samples = np.random.normal(m, std, gen_shape)
            samples = torch.from_numpy(samples).float()
        elif dist_type == 'uniform':
            samples = np.random.uniform(m, std, gen_shape)
            samples = torch.from_numpy(samples).float()
        elif dist_type == "ints":
            samples = np.random.randint(0, high=m, size=gen_shape)
            samples = torch.from_numpy(samples)
        elif dist_type == "hetero":
            samples = generate_heteroskedastic_ints(gen_shape, alpha, beta,m-1)
            samples = torch.from_numpy(samples)
        else:
            raise ValueError('dist_type must be normal, uniform, hetero, or ints.')

        
        ##after its trained a bit, it uses those weights to make "perfect" outputs
        data_loader = DataLoader(TensorDataset(samples), batch_size=batch_size, shuffle=False)
        self.data_loader = data_loader
        outputs_list = []
        inputs_list = []
        
        total_batches = len(data_loader)
        
        
        if store_outputs:
            if display_progress:
                progress_bar = tqdm(total=total_batches, desc=f"Generating {val_train} data :")
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("GPU is available")
            else:
                device = torch.device("cpu")
                print("GPU is not available, using CPU")
            
            self.model.to(device)
            self.model.eval()

            sample_batches = torch.split(samples, batch_size)


            outputs_list = []

            # Forward pass through the model in batches
            with torch.no_grad():  
                for batch in sample_batches:
                    batch = batch.to(device)
                    batch_outputs = self.model(batch)
                    outputs_list.append(batch_outputs)
                    if display_progress:
                        progress_bar.update(1)
            # Stack the batched outputs
            outputs_return = torch.cat(outputs_list, dim=0)

        if val_train == "train":
            self.train_inputs = samples  # Set your train inputs as needed
            if store_outputs:
                self.train_targets = outputs_return

        if val_train == "val":
            self.val_inputs = samples  # Set your val inputs as needed
            if store_outputs:
                self.val_targets = outputs_return 
    
    
    def graph_dataset_dist(self,val_train = 'val'):
        if val_train not in ["train","val"]:
            raise RuntimeError("please specify val_train = 'train' or 'val'.")

        if not self.cofigured:
            #if it is configured, we have self.input_shape and self.output shape
            raise RuntimeError("Teacher is not configured. Run the configure() method of your teacher object before plotting.")
        if val_train == "val":
            if self.val_targets.shape == torch.Size([0]):
                raise RuntimeError("There is no validation data to graph. set store_outputs = True in generate_data")
            test = torch.argmax(self.val_targets,axis = -1)
        else:
            if self.train_targets.shape == torch.Size([0]):
                raise RuntimeError("There is no training data to graph.  set store_outputs = True in generate_data")
            test = torch.argmax(self.train_targets,axis = -1)
        class_num = self.output_shape[0]
        class_counts = [np.count_nonzero(test == i) for i in range(1, class_num + 1)]

        # Create a bar chart
        plt.figure(figsize=(40, 10))
        plt.bar(range(1, class_num + 1), class_counts, align='center')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Bar Chart of Class Counts')
        plt.xticks(range(1, class_num + 1))

        plt.show()