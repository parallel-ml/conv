"""
    Write layer-wise configuration to json file store under system module.
"""
import json
from alexnet import original


PATH = 'system/resource/model/config.json'

with open('system/resource/model/config.json', 'w+') as f:
    model = original(include_fc=False)

    configs = dict()
    for layer in model.layers:
        layer_config = dict()
        layer_config['class_name'] = layer.__class__.__name__

        try:
            shape = layer.input_shape
        except Exception:
            shape = layer.get_input_shape_at(0)

        layer_config['input_shape'] = list(shape)[1:]
        layer_config['config'] = layer.get_config()
        configs[layer.get_config()['name']] = layer_config
    json.dump(configs, f)
