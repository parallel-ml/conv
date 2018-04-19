"""
    Write layer-wise configuration to json file store under system module.
"""
import json
import alexnet

PATH = 'system/resource/model/config.json'


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
    with open('system/resource/model/config.json', 'w+') as f:
        config = dict()
        model = alexnet.original(include_fc=False)
        save(model, f, config)
        model = alexnet.fc2()
        save(model, f, config)
        json.dump(config, f)


if __name__ == '__main__':
    main()
