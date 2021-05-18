import yaml
import collections.abc
import copy

def read_config_file(file_name):
    with open(file_name) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def save_config_file(file_name, data_dict):
    with open(file_name, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def config_updates(config,config_new):
	config_out = copy.deepcopy(config)
	for k,v in config_new.items():
		if isinstance(v,collections.abc.Mapping):
			config_out[k] = config_updates(config_out.get(k,{}),v)
		else:
			config_out[k] = v
	return config_out

if __name__=='__main__':
	_C = {}
	dataset = {}
	dataset['name'] = 'coco'
	dataset['root_path'] = 'COCO2017'
	dataset['train_path'] = 'val2017'
	dataset['valid_path'] = 'val2017'
	_C['dataset'] = dataset
	save_params = {}
	save_params['dir_save']        = os.path.join('results',dataset['name'])
	save_params['experiment_name'] = 'vgg16_bn_default'
	_C['save_params'] = save_params

	config_new = {}
	save_params = {}
	save_params['dir_save']        = '../'
	save_params['experiment_name'] = 'vgg16_bn'
	config_new['save_params'] = save_params
	config = config_updates(_C, config_new)
	print(config)
	print()

	config_new = {}
	config_new['new_params'] = '-----------'
	config = config_updates(_C, config_new)
	print(config)




















