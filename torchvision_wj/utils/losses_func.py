import torch
import torch.nn.functional as F
from .parallel_transform import parallel_transform


def mil_unary_sigmoid_loss(ypred, mask, gt_boxes, mode='all', 
                           focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                           epsilon=1e-6):
    """ Compute the mil unary loss.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]
    Returns
        unary loss for each category (C,) if mode='balance'
        otherwise, the average unary loss (1,) if mode='all'
    """
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob,0]
        c      = gt_boxes[nb_ob,1].item()
        box    = gt_boxes[nb_ob,2:]
        pred   = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        # print('***',c,box, pred.shape)
        if pred.numel() == 0:
            continue
        ypred_pos[c].append(torch.max(pred, dim=0)[0])
        ypred_pos[c].append(torch.max(pred, dim=1)[0])
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                bce_pos = -torch.log(torch.cat(ypred_pos[c], dim=0))
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_unary_approx_sigmoid_loss(ypred, mask, gt_boxes, mode='all', method='gm', gpower=4, 
                                  focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    if method=='gm':
        ypred_g = ypred**gpower
    elif method=='expsumr': #alpha-softmax function
        ypred_g = torch.exp(gpower*ypred)
    elif method=='explogs': #alpha-quasimax function
        ypred_g = torch.exp(gpower*ypred)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob,0]
        c      = gt_boxes[nb_ob,1].item()
        box    = gt_boxes[nb_ob,2:]
        pred   = ypred_g[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        if method=='gm':
            prob0 = torch.mean(pred, dim=0)**(1.0/gpower)
            prob1 = torch.mean(pred, dim=1)**(1.0/gpower)
        elif method=='expsumr':
            pd_org = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            prob0 = torch.sum(pd_org*pred,dim=0)/torch.sum(pred,dim=0)
            prob1 = torch.sum(pd_org*pred,dim=1)/torch.sum(pred,dim=1)
        elif method=='explogs':
            msk = mask[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            prob0 = torch.log(torch.sum(pred,dim=0))/gpower - torch.log(torch.sum(msk,dim=0))/gpower
            prob1 = torch.log(torch.sum(pred,dim=1))/gpower - torch.log(torch.sum(msk,dim=1))/gpower
        ypred_pos[c].append(prob0)
        ypred_pos[c].append(prob1)
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))
        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            neg = torch.clamp(1-ypred_neg[c], epsilon, 1-epsilon)
            bce_neg = -torch.log(neg)
            if len(ypred_pos[c])>0:
                pos = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pos)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_parallel_unary_sigmoid_loss(ypred, mask, crop_boxes, angle_params=(0,45,5), mode='all', 
                                    focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                                    obj_size=0, epsilon=1e-6):
    """ Compute the mil unary loss from parallel transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        crop_boxes: Tensor of boxes with (N, 5), where N is the number of bouding boxes in the batch,
                    the 5 elements of each row are [nb_img, class, center_x, center_r, radius]
    Returns
        polar unary loss for each category (C,) if mode='balance'
        otherwise, the average polar unary loss (1,) if mode='all'
    """
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        radius = ob_crop_boxes[nb_ob,-1]

        extra = 5
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]

        index = torch.nonzero(msk[0]>0.5, as_tuple=True)
        y0,y1 = index[0].min(), index[0].max()
        x0,x1 = index[1].min(), index[1].max()
        box_h = y1-y0+1
        box_w = x1-x0+1
        # print('-----',box_h,box_w, y1,y0)

        if min(box_h, box_w) <= obj_size:
            parallel_angle_params = [0]
        else:
            parallel_angle_params = list(range(angle_params[0],angle_params[1],angle_params[2]))
        # print("#angles = {}".format(len(parallel_angle_params)))

        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1  = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            pred_parallel0 = pred_parallel*msk0
            pred_parallel1 = pred_parallel*msk1
            flag0 = torch.sum(msk0[0], dim=0)>0.5
            prob0 = torch.max(pred_parallel0[0], dim=0)[0]
            prob0 = prob0[flag0]
            flag1 = torch.sum(msk1[0], dim=1)>0.5
            prob1 = torch.max(pred_parallel1[0], dim=1)[0]
            prob1 = prob1[flag1]
            if len(prob0)>0:
                ypred_pos[c.item()].append(prob0)
            if len(prob1)>0:
                ypred_pos[c.item()].append(prob1)
            # print(nb_ob,angle,len(prob0),len(prob1))
            # print(torch.unique(torch.sum(msk0[0], dim=0)))
            # print(torch.unique(torch.sum(msk1[0], dim=1)))
        #     plt.figure()
        #     plt.subplot(1,2,1)
        #     plt.imshow(msk0[0].cpu().numpy())
        #     plt.subplot(1,2,2)
        #     plt.imshow(msk1[0].cpu().numpy())
        #     plt.savefig('mask_'+str(angle)+'.png')
        # import sys
        # sys.exit()

    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_parallel_approx_sigmoid_loss(ypred, mask, crop_boxes, angle_params=(0,45,5), mode='all', 
                                     method='gm', gpower=4, 
                                     focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                                     obj_size=0, epsilon=1e-6):
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        radius = ob_crop_boxes[nb_ob,-1]

        extra = 5
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]

        index = torch.nonzero(msk[0]>0.5, as_tuple=True)
        y0,y1 = index[0].min(), index[0].max()
        x0,x1 = index[1].min(), index[1].max()
        box_h = y1-y0+1
        box_w = x1-x0+1

        if min(box_h, box_w) <= obj_size:
            parallel_angle_params = [0]
        else:
            parallel_angle_params = list(range(angle_params[0],angle_params[1],angle_params[2]))
        # print("parallel_angle_params: ", parallel_angle_params)

        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1  = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(msk0[0].cpu().numpy())
            # plt.subplot(1,2,2)
            # plt.imshow(msk1[0].cpu().numpy())
            # plt.savefig('mask_'+str(angle)+'.png')
            pred_parallel = pred_parallel[0]
            msk0 = msk0[0]>0.5
            msk1 = msk1[0]>0.5
            flag0 = torch.sum(msk0, dim=0)>0.5
            flag1 = torch.sum(msk1, dim=1)>0.5
            pred_parallel0 = pred_parallel[:,flag0]
            pred_parallel1 = pred_parallel[flag1,:]
            msk0 = msk0[:,flag0]
            msk1 = msk1[flag1,:]
            # plt.figure()
            # if torch.sum(flag0)>0.5:
            #     plt.subplot(1,2,1)
            #     plt.imshow(msk0.cpu().numpy())
            # if torch.sum(flag1)>0.5:
            #     plt.subplot(1,2,2)
            #     plt.imshow(msk1.cpu().numpy())
            # plt.savefig('mask_'+str(angle)+'_crop.png')
            
            if torch.sum(flag0)>0.5:
                if method=='gm':
                    w = pred_parallel0**gpower
                    prob0 = torch.sum(w*msk0, dim=0)/torch.sum(msk0, dim=0)
                    prob0 = prob0**(1.0/gpower)
                elif method=='expsumr':
                    w = torch.exp(gpower*pred_parallel0)
                    prob0 = torch.sum(pred_parallel0*w*msk0,dim=0)/torch.sum(w*msk0,dim=0)
                elif method=='explogs':
                    w = torch.exp(gpower*pred_parallel0)
                    prob0 = torch.log(torch.sum(w*msk0,dim=0))/gpower - torch.log(torch.sum(msk0, dim=0))/gpower
                ypred_pos[c.item()].append(prob0)
            if torch.sum(flag1)>0.5:
                if method=='gm':
                    w = pred_parallel1**gpower
                    prob1 = torch.sum(w*msk1, dim=1)/torch.sum(msk1, dim=1)
                    prob1 = prob1**(1.0/gpower)
                elif method=='expsumr':
                    w = torch.exp(gpower*pred_parallel1)
                    prob1 = torch.sum(pred_parallel1*w*msk1,dim=1)/torch.sum(w*msk1,dim=1)
                elif method=='explogs':
                    w = torch.exp(gpower*pred_parallel1)
                    prob1 = torch.log(torch.sum(w*msk1,dim=1))/gpower - torch.log(torch.sum(msk1,dim=1))/gpower
                ypred_pos[c.item()].append(prob1)
            # print(nb_ob,angle,len(prob0),len(prob1))
        # import sys
        # sys.exit()

    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_pairwise_loss(ypred, mask, softmax=True, exp_coef=-1):
    """ Compute the pair-wise loss.

        As defined in Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior

    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
    Returns
        pair-wise loss for each category (C,)
    """
    device = ypred.device
    center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),  
            torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]]),  
            torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])]
    ## pairwise loss for each col/row MIL
    num_classes = ypred.shape[1]
    if softmax:
        num_classes = num_classes - 1
    losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=device)
    for c in range(num_classes):
        pairwise_loss = []
        for w in pairwise_weights_list:
            weights = center_weight - w
            weights = weights.view(1, 1, 3, 3).to(device)
            aff_map = F.conv2d(ypred[:,c,:,:].unsqueeze(1), weights, padding=1)
            cur_loss = aff_map**2
            if exp_coef>0:
                cur_loss = torch.exp(exp_coef*cur_loss)-1
            cur_loss = torch.sum(cur_loss*mask[:,c,:,:].unsqueeze(1))/(torch.sum(mask[:,c,:,:]+1e-6))
            pairwise_loss.append(cur_loss)
        losses[c] = torch.mean(torch.stack(pairwise_loss))
    return losses