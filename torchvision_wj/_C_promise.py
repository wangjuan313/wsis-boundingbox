import os

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = {}
_C['device'] = 'cuda'

# -----------------------------------------------------------------------------
# dataset
# -----------------------------------------------------------------------------
dataset = {}
dataset['name'] = 'promise'
dataset['root_path'] = 'data/prostate/PROSTATE-Aug'
dataset['train_path'] = ('val/img','val/gt')#('train/img','train/gt')
dataset['valid_path'] = ('val/img','val/gt')
dataset['grp_regex']  = '(\\d+_Case\\d+_\\d+)_\\d+'
_C['dataset'] = dataset

# -----------------------------------------------------------------------------
# data parameters
# -----------------------------------------------------------------------------
data_params = {}
data_params['workers'] = 0
data_params['aspect_ratio_group_factor'] = 3
data_params['image_min_size']  = 256
data_params['image_max_size']  = 256
data_params['crop_width']      = 512
data_params['normalizer_mode'] = 'zscore'
data_params['crop_ob_prob']    = 0.75
data_params['crop_size']       = (256,256)
data_params['data_visual_aug'] = {}
data_params['data_visual_aug']['aug'] = False
data_params['data_visual_aug']['contrast_range']    = (0.9, 1.1)
data_params['data_visual_aug']['brightness_range']  = (-0.1, 0.1)
data_params['data_visual_aug']['hue_range']         = (-0.05, 0.05)
data_params['data_visual_aug']['saturation_range']  = (0.95, 1.05)
data_params['data_transform_aug'] = {}
data_params['data_transform_aug']['aug'] = False
data_params['data_transform_aug']['min_rotation']     = 0#-0.1
data_params['data_transform_aug']['max_rotation']     = 0#0.1
data_params['data_transform_aug']['min_translation']  = (-0.05, -0.05)
data_params['data_transform_aug']['max_translation']  = (0.05, 0.05)
data_params['data_transform_aug']['min_shear']        = 0#-0.1
data_params['data_transform_aug']['max_shear']        = 0#0.1
data_params['data_transform_aug']['min_scaling']      = (0.95, 0.95)
data_params['data_transform_aug']['max_scaling']      = (1.05, 1.05)
data_params['data_transform_aug']['flip_x_chance']    = 0.5
data_params['data_transform_aug']['flip_y_chance']    = 0.5
_C['data_params'] = data_params

# -----------------------------------------------------------------------------
# network parameters for retinanet
# -----------------------------------------------------------------------------
net_params = {}
net_params['input_dim']      = 1
net_params['model_name']     = 'unet_residual'
net_params['seg_num_classes'] = 1
net_params['softmax'] = False
net_params['losses'] = [('MILUnarySigmoidLoss',{'mode':'all', 'focal_params':{'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}},1),
						('MILPairwiseLoss',{'softmax':net_params['softmax'], 'exp_coef':-1},10)]
_C['net_params'] = net_params

# -----------------------------------------------------------------------------
# model training
# -----------------------------------------------------------------------------
train_params = {}
train_params['batch_size']   = 4
train_params['epochs']       = 50
train_params['start_epoch']  = 0
train_params['resume']       = ''
train_params['pretrained']   = False
train_params['test_only']    = False
train_params['lr']           = 1e-4
train_params['clipnorm']	 = 0.001
train_params['momentum']     = 0.9
train_params['weight_decay'] = 1e-4
train_params['lr_step_size'] = 8
train_params['lr_steps']     = [16, 22]
train_params['lr_gamma']     = 0.1
train_params['factor']       = 0.5
train_params['patience']     = 5
train_params['print_freq']   = 50
train_params['optimizer']    = 'Adam'#'SGD'
train_params['betas'] = (0.9, 0.99)
_C['train_params'] = train_params

# -----------------------------------------------------------------------------
# model saving
# -----------------------------------------------------------------------------
save_params = {}
save_params['dir_save_root']   = 'results'
save_params['dir_save']        = os.path.join('results',dataset['name'])
save_params['experiment_name'] = None
_C['save_params'] = save_params

