import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import uniform, normal
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

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


    
    def configure(self
                  , gen_lr = 0.01
                  , gen_epochs = 1000
                  , gen_init_range = (-1,1)
                  , gen_n = 10_000
                  , gen_m =0.0
                  , gen_std=1.0
                  , out_type = None
                  , batch_size = 50
                  , dist_type = 'normal'):
        
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
            samples = torch.from_numpy(np.random.randint(0, high=gen_m, size=gen_shape))
        else:
            raise ValueError('dist_type must be normal,uniform,or ints.')
        
        

        dataset = TensorDataset(samples, out_temp)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        progress_bar = tqdm(total=gen_epochs, desc="Configuring Teacher:")

        for epoch in range(gen_epochs):
            for batch_samples, batch_out_temp in dataloader:
                # Forward pass
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
        print("Teacher Configured")
    #this would be theoretical perfect dark knowledge
    def generate_data(self
                      , val_train = "train"
                      , n = 1000
                      , dist_type = 'normal'
                      , m =0.0
                      , batch_size = 50
                      , std=1.0
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
        else:
            raise ValueError('dist_type must be normal,uniform,or ints.')

        
        ##after its trained a bit, it uses those weights to make "perfect" outputs
        ####START HERE. need to make this batched
        outputs_return = self.model(samples)  
        
        if val_train == "train":
            self.train_inputs = samples #right now it is made to overwrite.  i could append?
            self.train_targets = outputs_return.detach() 
    
        if val_train == "val":
            self.val_inputs = samples #right now it is made to overwrite.  i could append?
            self.val_targets = outputs_return.detach() 
