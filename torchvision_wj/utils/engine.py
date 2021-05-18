import math
import sys
import time
import torch
from torchvision_wj.utils import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, clipnorm=0.001, print_freq=20):
    time.sleep(2)  # Prevent possible deadlock during epoch transition
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict, _ = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if clipnorm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def validate_loss(model, data_loader, device):
    n_threads = torch.get_num_threads()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation: '

    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    import pandas as pd
    loss_summary = []
    for images, targets in metric_logger.log_every(data_loader, print_freq=10e5, header=header, training=False):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict, _ = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_dict_reduced = dict(('val_'+k,f(v) if hasattr(v,'keys') else v) for k,v in loss_dict_reduced.items())

        loss_value = losses_reduced.item()
        metric_logger.update(val_loss=losses_reduced, **loss_dict_reduced)

        loss_reduced = dict((k,f(v) if hasattr(v,'keys') else v.item()) for k,v in loss_dict_reduced.items())
        loss_reduced.update(dict(gt=targets[0]["boxes"].shape[0]))
        loss_summary.append(loss_reduced)

    loss_summary = pd.DataFrame(loss_summary)
    loss_summary.to_csv("val_image_summary.csv", index=False)

    torch.set_num_threads(n_threads)
    
    return metric_logger

