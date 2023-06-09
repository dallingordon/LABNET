import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import uniform, normal
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim

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

class Teacher:

    def __init__(self,layer_sizes):
        
        # Extract the number of inputs and outputs from the layer_sizes list
        num_inputs = layer_sizes[0]
        num_outputs = layer_sizes[-1]
        
        self.model = nn.Sequential()
        self.inputs = torch.tensor([])
        self.targets = torch.tensor([])

        # Add input layer
        self.model.add_module("input_layer", nn.Linear(num_inputs, layer_sizes[1]))
        self.model.add_module("input_layer_activation", nn.ReLU())

        # Add hidden layers
        for i in range(2, len(layer_sizes)-1):
            self.model.add_module(f"hidden_layer_{i}", nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            self.model.add_module(f"hidden_layer_{i}_activation", nn.ReLU())

        # Add output layer
        self.model.add_module("output_layer", nn.Linear(layer_sizes[-2], num_outputs))




    #this would be theoretical perfect dark knowledge
    def generate_data(self
                      , n = 1000
                      , dist_type = 'normal'
                      ,  m =0.0
                      , std=1.0
                      , gen_lr = 0.01
                      , gen_epochs = 1000
                      , gen_init_range = (-1,1)
                     ):

        low,high = gen_init_range

        #initialize teh teacher weights
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                init.uniform_(module.weight, low, high)
                if module.bias is not None:
                    init.uniform_(module.bias, low, high)

        input_size = self.model.input_layer.in_features ##this only works with create_neural_network func above
        output_size = self.model.output_layer.out_features

        if dist_type == 'normal':
            samples = np.random.normal(m, std, (n,input_size))
        elif dist_type == 'uniform':
            samples = np.random.uniform(m, std, (n,input_size))
        else:
            raise ValueError('dist_type muste be either normal or uniform')

        samples = torch.from_numpy(samples).float()
        
        out_temp = np.random.normal(m, std, (n,output_size))
        out_temp = torch.from_numpy(out_temp).float()
        ##train for a few epochs to get decent weights
        ##random weights end up generating a lot of the same targets for random inputs.  its weird.  
        ##talk to kulis about this 
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=gen_lr)
        
        # Training loop
        for epoch in range(gen_epochs):
            # Forward pass
            outputs = self.model(samples)
            loss = criterion(outputs, out_temp)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        ##after its trained a bit, it uses those weights to make "perfect" outputs
        outputs_return = self.model(samples)  
        self.inputs = samples
        self.targets = outputs_return.detach()
    

class Neuron:
    def __init__(self, neuron_type):
        """
        'i' is input
        'o' is output
        'fc' is fully connected
        """
        self.type = neuron_type 
        self.connections = []
        self.output = False

    def add_connections(self, neuron_ids):
        if isinstance(neuron_ids, int):
            self.connections.append(neuron_ids)
        elif isinstance(neuron_ids, list):
            self.connections.extend(neuron_ids)
        else:
            raise ValueError("Invalid input. Expected an integer or a list of integers.")


    def remove_connections(self, neuron_ids):
        if isinstance(neuron_ids, int):
            if neuron_ids in self.connections:
                self.connections.remove(neuron_ids)
        elif isinstance(neuron_ids, list):
            self.connections = [neuron for neuron in self.connections if neuron not in neuron_ids]
        else:
            raise ValueError("Invalid input. Expected an integer or a list of integers.")

    def get_connections(self):
        return self.connections
    
    def type(self):
        return self.type
    
    def set_output(self,val):
        self.output = val
        
    def get_output(self):
        return self.output
   


class Net:
    def __init__(self,input_size, hidden_size, output_size):
        self.neurons = []
        self.input_size = input_size
        self.output_size = output_size
        self.inputs = []
        self.outputs = []
        for i in range(input_size):
            self.neurons.append(Neuron('i'))
            
        for i in range(hidden_size):
            self.neurons.append(Neuron('fc'))
            
        for i in range(output_size):
            self.neurons.append(Neuron('o'))
        self.size = len(self.neurons)
        
    def __init__(self,design_list):
        """
        design_list is a list of lists:
        [
            ['i',[]],
            ['fc',[1,-3,4]],
            ['o',[1,2]]


        ]
        """
        self.neurons = []
        self.input_size = 0
        self.output_size = 0
        self.inputs = []
        self.outputs = []


        for i in design_list:
            if i[0] == 'i':
                self.neurons.append(Neuron('i'))
                self.input_size += 1

            if i[0] == 'o':
                this_neuron = Neuron('o')
                this_neuron.add_connections(i[1])
                self.neurons.append(this_neuron)
                self.output_size += 1

            if i[0] == 'fc':
                this_neuron = Neuron('fc')
                this_neuron.add_connections(i[1])
                self.neurons.append(this_neuron)

        self.size = len(self.neurons)
    
    def get_neurons(self):
        return self.neurons

    def get_neuron_at(self,idx):
        return self.neurons[idx]
    
    def get_output_at(self,idx):
        return self.neurons[idx].get_output()
    
    def get_output(self,input_data):
        
        final_res = []
        assert len(input_data) == self.input_size
        
        for idx,n in enumerate(self.neurons):
            if n.type == 'i':
                n.set_output(input_data[idx])
                
            if n.type in ['fc','o']:
                res = True
                for i in n.get_connections():
                    if i < 0:
                        ###tiny thing, no such thing as negative 0 so i can't do the not on the first input.  I don't care
                        res = res and not(self.get_output_at(-1*i))
                    else:
                        res = res and self.get_output_at(i)
                n.set_output(res)
                if n.type == 'o':
                    final_res.append(res)
        return final_res
    
    def generate_inputs(self, num_bits = None):
        if num_bits is None:
            num_bits = self.input_size
            
        if num_bits <= 0:
            self.inputs = [[]]
            return

        self.generate_inputs(num_bits - 1)
        prev_combinations = self.inputs
        combinations = []
        for combination in prev_combinations:
            combinations.append(combination + [True])
            combinations.append(combination + [False])

        self.inputs = combinations
        
        #no return, they are all in self.inputs
    def generate_outputs(self):
        assert len(self.inputs) > 0, "You must run generate_inputs() before generate_outputs.  duh."
        self.outputs = [self.get_output(i) for i in self.inputs]
    
    def percent_true(self):
        assert len(self.outputs) > 0, "You must run generate_outputs() before percent_true.  duh."
        res = [0] * self.output_size
        n = len(self.outputs)
        for i in self.outputs:
            for idx, value in enumerate(i):
                res[idx] += value
        return [i/n for i in res]
                
    def tensorize(self):
        assert len(self.inputs) > 0, "You must run generate_inputs() before tensorizing.  duh."
        assert len(self.outputs) > 0, "You must run generate_outputs() before tensorizing.  duh."
        self.inputs = torch.tensor(self.inputs)
        self.outputs = torch.tensor(self.outputs)
        print("self.inputs and self.outputs are now tensors of shape {} and {} respectively!".format(self.inputs.shape, self.outputs.shape))
  

class Lab:
    def __init__(self, model,num_epochs,samples):
        """
        used in the training loop to record all the weights.  this will do differencing and visualizations
        """
        self.LabParams = {} #holds pretty much everything
        self.LayerNames = []
        
        weight_layers = sum(1 for i in model.named_parameters())
        #self.LabParams = [0]*weight_layers # a list containing the weight tensors
        self.LayerNames = ['']*weight_layers
        ep_list = [num_epochs* samples]

        for idx,t in enumerate(model.named_parameters()):
            name = t[0]
            layer_shape = ep_list + list(t[1].shape) #the shape the lab param should have for each weight matrix
            self.LabParams[name] = torch.zeros(layer_shape)
            self.LayerNames[idx] = name
        
        
    def record(self, model,epoch,data_samples,sample):
        #this has to know which weight layer it is.  i think its always the same order?
        idx = 0
        for i in model.named_parameters():
            #assert i[0] == self.LabParams[idx][0] 
            self.LabParams[i[0]][epoch * data_samples + sample] = i[1]
            idx +=1
    def graph(self
              ,layers_to_graph = None
              , graph_together = False
              , diff = 0
              ,plot_size = (10,6)
              ,x_range = (None, None)
              ,y_range = (None, None)
             ):
        
        if layers_to_graph is None:
            layers_to_graph = self.LayerNames

        if graph_together:
            fig, ax = plt.subplots(figsize=plot_size)
            

        for layer_name in layers_to_graph:
        #layer_name = 'hidden_1.weight'


            weights = self.LabParams[layer_name].detach().numpy()
            shapes =  (weights.shape[0], np.prod(weights.shape[1:])) #flattens all but first (time step)
            weights = weights.reshape(shapes)
            
            for i in range(diff): ##this diffs to the derivative you want.  1 is first, 2 2nd, etc.
                weights = np.diff(weights, axis=0)
                
                #shapes = weights.shape
                
            num_time_steps, num_dimensions = weights.shape

            if not graph_together:
                fig, ax = plt.subplots(figsize=plot_size)

            for i in range(num_dimensions):
                ax.plot(range(num_time_steps), weights[:, i], label=f"Dimension {i+1}")


            ax.set_xlabel('Time Step')
            ax.set_ylabel('Weight Value')
            ax.set_title(layer_name)
            if not graph_together:
                if x_range[0] is not None and x_range[1] is not None:
                    plt.xlim(x_range[0], x_range[1])
                if y_range[0] is not None and y_range[1] is not None:
                    plt.ylim(y_range[0], y_range[1])
                
                plt.show()

        if graph_together:
            if x_range[0] is not None and x_range[1] is not None:
                plt.xlim(x_range[0], x_range[1])
            if y_range[0] is not None and y_range[1] is not None:
                plt.ylim(y_range[0], y_range[1])
            
            plt.show()
         
    

