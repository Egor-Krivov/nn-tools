import numpy as np
import pandas as pd
from lasagne.layers import get_all_layers


def count_layer_params(layer, **tags):
    """Count number of parameters for lasagne layer."""
    n_params = 0
    for p in layer.get_params(**tags):
        shape = p.get_value().shape
        n_params += np.prod(shape)
    return n_params


def make_net(output):
    """Form dictionary from incoming layers of lasagne output. These layers need
    to have names."""
    net = {}
    for l in get_all_layers(output):
        name = l.name
        if l.name is not None:
            net[name] = l
    return net


def summary(output):
    """Form dataframe from lasagne output."""
    layer_info = []
    for l in get_all_layers(output):
        layer_type = l.__class__.__name__
        name = l.name
        shape = l.output_shape
        params = [p.get_value().shape for p in l.get_params()]
        params = params if len(params) else None
        params_total = count_layer_params(l)
        layer_info.append((layer_type, name, shape, params, params_total))

    d = pd.DataFrame(layer_info,
                     columns=['layer_type', 'name', 'shape', 'params',
                              'params_total'])


    return pd.DataFrame(layer_info,
                    columns=['layer_type', 'name', 'shape', 'params',
                             'params_total'])