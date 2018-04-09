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
        config = layer.get_config()
        configs[config['name']] = config
    json.dump(configs, f)
