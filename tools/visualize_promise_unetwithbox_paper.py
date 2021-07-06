import os, sys
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import torch
from torchvision_wj.datasets.samplers import PatientSampler
from torchvision_wj.models.segwithbox.unetwithbox import UNetWithBox
from torchvision_wj.models.segwithbox.default_unet_net import *
from torchvision_wj.utils.losses import *
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.promise_utils import get_promise
import torchvision_wj.utils.transforms as T
import cv2

def get_hyper_parameters(dice_2d_all):
    epochs = list(dice_2d_all.keys())
    threshold = list(dice_2d_all[epochs[0]].keys())
    assert len(epochs)==50, len(epochs)
    mean_2d_array, std_2d_array = [], []
    for key in dice_2d_all.keys():
        mean_2d_array.append(np.mean(np.asarray(dice_2d_all[key]),axis=0))
        std_2d_array.append(np.std(np.asarray(dice_2d_all[key]),axis=0))
    mean_2d_array = np.vstack(mean_2d_array)
    std_2d_array = np.vstack(std_2d_array)
    
    ind = np.where(mean_2d_array==np.max(mean_2d_array))
    epoch = epochs[ind[0][0]]
    th = threshold[ind[1][0]]
    dice_summary = dice_2d_all[epoch][th]
    return int(epoch), th, dice_summary

@torch.no_grad()
def evaluate(model, data_loader, image_names, device, threshold, 
             save_detection=None, smooth=1e-10):
    # torch.set_num_threads(1)
    model.eval()
    print(dice_summary)
    nb_img = 0
    for images, targets in data_loader:
        nb_img += 1
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        image_ids = np.array([t["image_id"].item() for t in targets], dtype=int)
        _, outputs = model(images, targets)
        out = outputs[0]
        
        gt_masks = torch.stack([t["masks"] for t in targets], dim=0).bool()
        pred = out>threshold
        pred = pred[:gt_masks.shape[0],:gt_masks.shape[1],:gt_masks.shape[2],:gt_masks.shape[3]]

        # dice coefficient
        smooth = 1e-10
        dice_3d = (2*(pred&gt_masks).sum()+smooth)/(pred.sum()+gt_masks.sum()+smooth)
        dice_2d = 2*(pred&gt_masks).sum(dim=(1,2,3))/(pred.sum(dim=(1,2,3))+gt_masks.sum(dim=(1,2,3))+smooth)
        # dice_3d = dice_3d.item()
        dice_2d = dice_2d.cpu().numpy()

        assert len(image_ids) == gt_masks.shape[0]
        ind_select = np.argsort(dice_2d)[::-1]
        ind_select = ind_select[:8]
        print(dice_3d, dice_2d[ind_select])
        alpha = 0.4
        for k in ind_select:
            img_id = image_ids[k]
            print(f"image = {nb_img}, slice = {k}, img_id = {img_id}")
            img = 255 * (images[k] - images[k].min())/(images[k].max()-images[k].min())
            img = img[0].cpu().numpy().astype(np.uint8)
            mask = (gt_masks[k,0]*255).cpu().numpy().astype(np.uint8)
            gt_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            gt_mask[:,:,0] = mask
            gt_img = [img, img, img]
            gt_img = np.stack(gt_img, axis=2)
            gt_img = cv2.addWeighted(gt_mask, alpha, gt_img, 1 - alpha, 0)
            pd = (pred[k,0]*255).cpu().numpy().astype(np.uint8)
            pd_mask = np.zeros((pd.shape[0], pd.shape[1], 3), dtype=np.uint8)
            pd_mask[:,:,0] = pd
            pd_img = [img, img, img]
            pd_img = np.stack(pd_img, axis=2)
            pd_img = cv2.addWeighted(pd_mask, alpha, pd_img, 1 - alpha, 0)
            cv2.imwrite(os.path.join(save_detection, f"gt_{nb_img}_{k}_{img_id}.png"), gt_img)
            cv2.imwrite(os.path.join(save_detection, f"pd_{nb_img}_{k}_{img_id}.png"), pd_img)
        
        
if __name__ == "__main__":
    dir_save_root = os.path.join('results','promise')
    folders = ['residual_parallel_approx_focal_40_20_expsumr=4_unary_pair',
               'residual_parallel_approx_focal_40_20_explogs=6_unary_pair']
  
    for experiment_name in folders:
        output_dir = os.path.join(dir_save_root, experiment_name)
        file_name = os.path.join(output_dir, 'dice_3d.xlsx')
        results = pd.read_excel(file_name, sheet_name=None)
        epoch, threshold, dice_summary = get_hyper_parameters(results)
        print(experiment_name, epoch, threshold)
        
        output_dir = os.path.join(dir_save_root, experiment_name)
        _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))
        assert _C['save_params']['experiment_name']==experiment_name, "experiment_name is not right"
        cfg = {'data_params': {'workers': 0}}
        _C = config.config_updates(_C, cfg)

        train_params       = _C['train_params']
        data_params        = _C['data_params']
        net_params         = _C['net_params']
        dataset_params     = _C['dataset']
        save_params        = _C['save_params']
        print('workers = %d' % data_params['workers'])

        device = torch.device(_C['device'])

        def get_transform():
            transforms = []
            transforms.append(T.ToTensor())
            transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
            return T.Compose(transforms)

        # Data loading code
        print("Loading data")
        dataset_test = get_promise(root=dataset_params['root_path'], 
                                   image_folder=dataset_params['valid_path'][0], 
                                   gt_folder=dataset_params['valid_path'][1], 
                                   transforms=get_transform(),
                                   transform_generator=None, visual_effect_generator=None)
        image_names = dataset_test.image_names
        
        print("Creating data loaders")
        test_patient_sampler = PatientSampler(dataset_test, dataset_params['grp_regex'], shuffle=False)

        data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1,
                batch_sampler=test_patient_sampler, num_workers=data_params['workers'],
                collate_fn=utils.collate_fn, pin_memory=True)

        print("Creating model with parameters: {}".format(net_params))
        
        net = eval(net_params['model_name'])(net_params['input_dim'], net_params['seg_num_classes'],
                                             net_params['softmax'])
        losses, loss_weights = [], []
        for loss in net_params['losses']:
            losses.append(eval(loss[0])(**loss[1]))
            loss_weights.append(loss[2])
        model = UNetWithBox(net, losses, loss_weights, softmax=net_params['softmax'])
        model.to(device)
        
        model_file = 'model_{:02d}'.format(epoch)
        print('loading model {}.pth'.format(model_file))
        checkpoint = torch.load(os.path.join(output_dir, model_file+'.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
        print('start evaluating {} ...'.format(epoch))
        model.training = False
        save_detection = os.path.join(output_dir, f'visualization_epoch={epoch}_T={threshold}')
        os.makedirs(save_detection, exist_ok=True)
        evaluate(model, data_loader_test, image_names=image_names, device=device, 
                 threshold=threshold, save_detection=save_detection)