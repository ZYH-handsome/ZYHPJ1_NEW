import pickle

from debugpy.launcher import channel

from .layers import *
from abc import abstractmethod


class NeuralNetworkModel:
    def __init__(self):
        self.layer_list = []

    @abstractmethod
    def forward(self, input: np.ndarray):
        pass

    @abstractmethod
    def backward(self, grads: np.ndarray):
        pass

    @abstractmethod
    def load_model(self, save_path):
        pass

    @abstractmethod
    def save_model(self, save_path):
        pass

class MLPModel(NeuralNetworkModel):
    def __init__(self, size_list: list[int]=None, act_func:str=None, **kwargs):
        super().__init__()
        self.size_list = size_list
        self.layer_list = []
        self.act_func = act_func
        self.act_func_map = {'ReLU': ReLU, 'Sigmoid': Sigmoid, 'LeakyReLU': LeakyReLU}
        self.weight_decay_list = kwargs.get('weight_decay_list')
        assert not self.weight_decay_list or len(self.weight_decay_list) == len(self.size_list) - 1, 'weight decay doesn\'t match'

        self.save_dir = kwargs.get('save_dir')

        if self.save_dir:
            self.load_model(self.save_dir)
        else:
            for i in range(len(size_list) - 1):
                layer = Linear(self.size_list[i], self.size_list[i+1])
                # add weight decay
                if self.weight_decay_list and layer.weight_decay:
                    layer.weight_decay_param = self.weight_decay_list[i]
                self.layer_list.append(layer)
                # add activate layer(except the last layer)
                if i < len(size_list) - 2:
                    try:
                        self.layer_list.append(self.act_func_map[self.act_func]())
                    except ValueError:
                        print('activate function not been finished')

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, input: np.ndarray):
        assert self.size_list and self.act_func, 'Model has not been correctly initialized, try using load_model method or directly provide size_list and act_func'
        for layer in self.layer_list:
            input = layer(input)
        return input

    def backward(self, grads: np.ndarray):
        for layer in reversed(self.layer_list):
            grads = layer.backward(grads)

    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            param_list = pickle.load(f)

        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            layer = Linear(self.size_list[i], self.size_list[i+1])
            layer.params['W'] = param_list[i + 2]['W']
            layer.params['b'] = param_list[i + 2]['b']
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_param = param_list[i + 2]['weight_decay_param']
            self.layer_list.append(layer)
            if i < len(self.size_list) - 2:
                self.layer_list.append(self.act_func_map[self.act_func]())

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layer_list:
            if type(layer) not in self.act_func_map.values():
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'weight_decay_param': layer.weight_decay_param
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

class CNN(NeuralNetworkModel):
    def __init__(self, channel_list: list[int]=None, kernel_list: list[tuple[int]]=None, linear_size_list: list[int]=None, act_func:str=None, **kwargs):
        super().__init__()
        self.channel_list = channel_list
        self.kernel_list = kernel_list
        self.linear_size_list = linear_size_list
        self.layer_list = []

        assert (not self.channel_list and not self.kernel_list) or len(self.channel_list) == len(self.kernel_list) + 1, '[Error]length of channel list doesn\'t match that of kernel_list'

        self.act_func = act_func
        self.act_func_map = {'ReLU': ReLU, 'Sigmoid': Sigmoid, 'LeakyReLU': LeakyReLU}
        self.weight_decay_list = kwargs.get('weight_decay_list')
        self.linear_weight_decay_list = kwargs.get('linear_weight_decay_list')

        assert not self.weight_decay_list or len(self.weight_decay_list) == len(self.channel_list) - 1, '[Error]Weight decay doesn\'t match'
        assert not self.linear_weight_decay_list or len(self.linear_weight_decay_list) == len(self.linear_size_list) - 1, '[Error]Linear weight decay doesn\'t match'

        self.save_dir = kwargs.get('save_dir')

        # init model
        if self.save_dir:
            self.load_model(self.save_dir)
        else:
            for i in range(len(self.channel_list) - 1):
                layer = Conv2D(in_channel=self.channel_list[i], out_channel=self.channel_list[i+1], kernel_size=self.kernel_list[i])
                # add weight decay
                if self.weight_decay_list and layer.weight_decay:
                    layer.weight_decay_param = self.weight_decay_list[i]
                self.layer_list.append(layer)

                # add activation function
                self.layer_list.append(self.act_func_map[self.act_func]())

                # add pooling layer
                self.layer_list.append(MaxPooling())

            # add flatten layer
            self.layer_list.append(Flatten())

            # add linear layers
            for i in range(len(self.linear_size_list) - 1):
                layer = Linear(self.linear_size_list[i], self.linear_size_list[i+1])
                if self.linear_weight_decay_list and layer.weight_decay:
                    layer.weight_decay_param = self.linear_weight_decay_list[i]
                self.layer_list.append(layer)

                # add activation function
                if i < len(self.linear_size_list) - 2:
                    self.layer_list.append(self.act_func_map[self.act_func]())

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, input: np.ndarray):
        assert self.channel_list and self.kernel_list and self.act_func, 'Model has not been correctly initialized, try using load_model method or directly provide channel_list, kernel_list and act_func'
        for layer in self.layer_list:
            # print(f'[Forward]Layer:{type(layer)}')
            input = layer(input)
        return input

    def backward(self, grads: np.ndarray):
        for layer in reversed(self.layer_list):
            # print(f'[Backward]Layer:{type(layer)}')
            grads = layer.backward(grads)

    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            param_list = pickle.load(f)

        self.channel_list = param_list[0]
        self.kernel_list = param_list[1]
        self.linear_size_list = param_list[2]
        self.act_func = param_list[3]

        for i in range(len(self.channel_list) - 1):
            # add Conv2D layer
            layer = Conv2D(self.channel_list[i], self.channel_list[i + 1], self.kernel_list[i])
            layer.params['W'] = param_list[i + 4]['W']
            layer.params['b'] = param_list[i + 4]['b']
            layer.weight_decay = param_list[i + 4]['weight_decay']
            layer.weight_decay_param = param_list[i + 4]['weight_decay_param']
            self.layer_list.append(layer)

            # add activation function
            self.layer_list.append(self.act_func_map[self.act_func]())

            # add pooling layer
            self.layer_list.append(MaxPooling())

        # add flatten layer
        self.layer_list.append(Flatten())

        # add ffn
        for i in range(len(self.linear_size_list) - 1):
            layer = Linear(self.linear_size_list[i], self.linear_size_list[i+1])
            layer.params['W'] = param_list[i + len(self.kernel_list) + 4]['W']
            layer.params['b'] = param_list[i + len(self.kernel_list) + 4]['b']
            layer.weight_decay = param_list[i + len(self.kernel_list) + 4]['weight_decay']
            layer.weight_decay_param = param_list[i + len(self.kernel_list) + 4]['weight_decay_param']
            self.layer_list.append(layer)

            # add activation function
            if i < len(self.linear_size_list) - 2:
                self.layer_list.append(self.act_func_map[self.act_func]())

    def save_model(self, save_path):
        param_list = [self.channel_list, self.kernel_list, self.linear_size_list, self.act_func]
        for layer in self.layer_list:
            if isinstance(layer, Conv2D):
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'weight_decay_param': layer.weight_decay_param
                })
            elif isinstance(layer, Linear):
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'weight_decay_param': layer.weight_decay_param
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def print_struct(self):
        print("[Dev]The structure of CNN:")
        for layer in self.layer_list[:-1]:
            if isinstance(layer, Conv2D):
                print(f"Conv2D: in_channels:{layer.params['W'].shape[0]}, out_channels:{layer.params['W'].shape[1]}, "
                      f"size:{layer.params['W'].shape[2:]}")
                print(f"MaxPooling")
                print(f"Activation function:{self.act_func}")
            elif isinstance(layer, Linear):
                print(f"Linear: in_dim:{layer.params['W'].shape[0]}, out_dim:{layer.params['W'].shape[1]}")
                print(f"Activation function:{self.act_func}")
        last_layer = self.layer_list[-1]
        print(f"Linear: in_dim:{last_layer.params['W'].shape[0]}, out_dim:{last_layer.params['W'].shape[1]}")