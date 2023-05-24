import torch

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
        
