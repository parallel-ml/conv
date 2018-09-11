"""
    Write layer-wise configuration to json file store under system module.
"""
import json
import alexnet
import os


PWD = os.environ['PWD']


def save(model, f, config):
    for layer in model.layers:
        if layer.__class__.__name__ == 'Concatenate':
            continue

        layer_config = dict()
        layer_config['class_name'] = layer.__class__.__name__

        try:
            shape = layer.input_shape
        except Exception:
            shape = layer.get_input_shape_at(0)

        layer_config['input_shape'] = list(shape)[1:]
        layer_config['config'] = layer.get_config()
        config[layer.get_config()['name']] = layer_config


def main():
    path = 'system/resource/model/alexnet/1/config.json'
    with open(path, 'w+') as f:
        config = dict()
        model = alexnet.original(include_fc=False)
        save(model, f, config)
        json.dump(config, f)

    path = 'system/resource/model/alexnet_filter/1/config.json'
    with open(path, 'w+') as f:
        config = dict()
        model = alexnet.filter(include_fc=False)
        save(model, f, config)
        json.dump(config, f)

    path = 'system/resource/model/alexnet_xy/1/config.json'
    with open(path, 'w+') as f:
        config = dict()
        model = alexnet.xy(include_fc=False)
        save(model, f, config)
        json.dump(config, f)

    path = 'system/resource/model/alexnet_channel/1/config.json'
    with open(path, 'w+') as f:
        config = dict()
        model = alexnet.channel(include_fc=False)
        save(model, f, config)
        json.dump(config, f)


if __name__ == '__main__':
    main()
