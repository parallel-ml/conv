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
        layer_config['config'] = layer.get_config()
        configs[layer.get_config()['name']] = layer_config
    json.dump(configs, f)
