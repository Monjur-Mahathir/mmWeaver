import torch
import torch.nn as nn
from src.utils import *

class INRBlock(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        self.first_layer_init = None 

        nonlinearities_inits = {
            'sine': (Sine(), sine_init, first_layer_sine_init),
            'relu': (nn.ReLU(inplace=True), init_weights_relu, None),
            'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
            'tanh':(nn.Tanh(), init_weights_xavier, None),
            'selu':(nn.SELU(inplace=True), init_weights_selu, None),
            'elu':(nn.ELU(inplace=True), init_weights_elu, None)
        }

        non_linearity, non_linearity_init, first_layer_init = nonlinearities_inits[nonlinearity]

        if weight_init is not None:
            self.weight_init = weight_init
        else:
            self.weight_init = non_linearity_init

        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            non_linearity
        ))

        for i in range(hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(in_features=hidden_features, out_features=hidden_features),
                non_linearity
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(
                nn.Linear(in_features=hidden_features, out_features=out_features)
            ))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(in_features=hidden_features, out_features=out_features),
                non_linearity
            ))

        self.net = nn.Sequential(*self.net)

        if self.weight_init is not None:
            self.net.apply(self.weight_init)
        
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)
    
    @property
    def flops(self):
        return (self.in_features + 1) * self.hidden_features + self.hidden_layers * (self.hidden_features+1) * self.hidden_features + \
               (self.hidden_features + 1) * self.out_features

    def forward(self, encoded_coordinates):
        output = self.net(encoded_coordinates)
        return output


class INRNet(nn.Module):
    def __init__(self, positional_embedding='ffm', in_features=4, out_features=2, 
                 hidden_layers=4, hidden_features=256, nonlinearity='relu', **kwargs):
        super().__init__()
        self.positional_embedding = positional_embedding

        if self.positional_embedding == 'pe':
            self.map = PositionalEncoding(in_features=in_features, num_frequencies=kwargs.get('num_frequencies'),
                                          sidelength=kwargs.get('sidelength', None),
                                          use_nyquist=kwargs.get('use_nyquist', False))
        elif self.positional_embedding == 'ffm':
            self.map = FourierFeatMapping(in_dim=in_features, map_scale=kwargs.get('map_scale', 128), map_size=kwargs.get('map_size', 4096))
        else:
            raise ValueError(f'Unknown type of positional embedding: {self.positional_embedding}')
        
        self.net = INRBlock(in_features=self.map.out_dim, out_features=out_features, hidden_layers=hidden_layers,
                            hidden_features=hidden_features, outermost_linear=True, nonlinearity=nonlinearity)
    
    @property
    def flops(self):
        return self.map.flops + self.net.flops

    def forward(self, coords):
        coords = coords.clone().requires_grad_(True)
        encoded_coordinates = self.map(coords)

        output = self.net(encoded_coordinates)
        return output
