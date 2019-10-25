from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import os
import torch as t
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from config import opt
import time
from Data.dataset import Dataset, TestDataset, inverse_normalize
from torchvision.models import vgg16
from torch.utils import data as data_
from collections import namedtuple
from Utils import array_tool as at
from Utils.vis_tool import visdom_bbox
from Utils.eval_tool import eval_detection_voc
from Utils.roi_module  import RoIPooling2D
import my_generator as gtor
from torchnet.meter import ConfusionMeter, AverageValueMeter
import cupy as cp
from Utils.nms import non_maximum_suppression
import my_bbox_tool  as bt
from Utils.vis_tool import *
import torch.optim as optim
import ipdb
from Data.dataset import preprocess
import cv2
from collections import namedtuple
from PIL import Image
import xml.etree.cElementTree as ET


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss



def decom_vgg16():
    if opt.caffe_pretrain:
        model = vgg16(pretrained = False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    #freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier

class rpn(nn.Module):
    """
    out put roi, roiindices, 2 branches of prediction for rpn and all anchors(used for genearting anchortarget),propasal
    输出roi，roiindices，rpn的2路预测，和所有的anchor（用来后面生成anchortarget），propalsal的生成
    """
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16):
        super(rpn, self).__init__()
        self.anchor_base = bt.base_anchor_generator(scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, kernel_size=1, padding=0, stride=1)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, kernel_size=1, padding=0, stride=1)

        # initialize
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.constant_(self.score.bias, 0.1)
        nn.init.xavier_uniform_(self.loc.weight)
        nn.init.constant_(self.loc.bias, 0.1)

    def forward(self, x,img_size,n_pre_nms,n_post_nms,scale):
        n, _, H, W = x.shape
        anchor = bt.enumerate_shift_anchor(np.array(self.anchor_base),self.feat_stride, H, W)
        n_anchor = anchor.shape[0] // (H * W)

        h = F.relu(self.conv1(x))

        rpn_loc = self.loc(h)
        # change to N,H,C,W,then turned into form of coordinate, shoulbe (9*H*W,4)
        # 转成N,H,W,C,然后转成坐标形式，应该是（9*H*W，4）
        # score and loc both need reshape, since the input for propasalcreator should be 2-dimensinal
        #score和loc都要reshape一下，因为输入进propasalcreator应该是是二维的
        rpn_loc = rpn_loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_score = self.score(h)

        # turn to N,H,W,C
        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous()

        rpn_softmax_score = F.softmax(rpn_score.view(n, H, W, n_anchor, 2), dim=4)
        # the second channel the probability of foreground
        # 第二个通道是前景概率
        rpn_fg_score = rpn_softmax_score[:, :, :, :, 1].contiguous()
        # turn to 2-dimesional 变成二维
        rpn_fg_score = rpn_fg_score.view(n, -1)
        # 三维（batchsize，H*W,channel)
        rpn_score = rpn_score.view(n, -1, 2)

        # generate roi, here uses roiindices because we use the roipooling from others
        #生成roi，这里要roiindices因为用了其他人写的roipooling
        rois = list()
        roi_indices = list()
        ProposalCreator = gtor.ProposalCreator(
            img_size, n_pre_nms, n_post_nms,
            anchor,scale
        )
        for i in range(n):
            roi = ProposalCreator(rpn_loc[i].cpu().data.numpy(),rpn_fg_score[i].cpu().data.numpy())
            batch_index = i * np.ones((len(roi[0:]),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        return rpn_loc, rpn_score, rois, roi_indices, anchor

class roihead(nn.Module):
    def __init__(self,n_class,roi_size,spatial_scale,classifier):
        #n_class include the bg
        super(roihead,self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        nn.init.xavier_uniform_(self.cls_loc.weight)
        nn.init.constant_(self.cls_loc.bias, 0)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.constant_(self.score.bias, 0)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        H,W = self.roi_size
        self.roi = RoIPooling2D(H,W,self.spatial_scale)

    def forward(self,x,rois,roi_indices):
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:,None],rois],dim = 1)

        #yx -> xy
        xy_indices_and_rois = indices_and_rois[:,[0,2,1,4,3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x,indices_and_rois)
        # unfold, then connect FC layer
        #展开，后面接FC
        pool = pool.view(pool.size(0),-1)
        fc7 = self.classifier(pool)
        roi_cls_loc = self.cls_loc(fc7)
        roi_score = self.score(fc7)
        return roi_score,roi_cls_loc


LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class fasterrcnn(nn.Module):
    def __init__(self,n_class,roi_size,spatial_scale):
        super(fasterrcnn,self).__init__()
        self.n_class = n_class
        self.extractor,self.classifier = decom_vgg16()
        self.rpn= rpn()
        self.roihead = roihead(n_class,roi_size,spatial_scale,self.classifier)

        # mean and std
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.use_preset('evaluate')

    def forward(self,imgs,n_pre_nms,n_post_nms,scale):

        _,_,H,W = imgs.shape

        img_size = (H,W)

        feature = self.extractor(imgs)

        rpn_loc,rpn_cls,roi,roi_indices,anchor = self.rpn(feature,img_size,n_pre_nms,n_post_nms,scale)

        roi_cls, roi_loc = self.roihead(feature,roi,roi_indices)

        return roi_loc,roi_cls,roi,roi_indices

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            #(N,one class,4) = (N,1,4)
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            #(N,one lcass） = (N,1)
            prob_l = raw_prob[:, l]
            #(K,)
            mask = prob_l > self.score_thresh
            #select top K whose score is higher than threshold from N
            #从N个里面，选出得分高于阈值的K个  
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            # nms
            #选出来后进行nms
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            # keep nms
            #保留nms的框
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def predict(self, imgs, sizes=None, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img,opt.test_n_pre_nms,opt.test_n_post_nms,scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = bt.loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

class fasterrcnn_train(nn.Module):
    def __init__(self,faster_rcnn):
        super(fasterrcnn_train, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchortarget = gtor.Anchortarget_generator()
        self.propasaltarget = gtor.ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self,imgs,bbox,label,n_pre_nms,n_post_nms,scale):

        _,_,H,W = imgs.shape

        img_size = (H,W)

        feature = self.faster_rcnn.extractor(imgs)

        rpn_loc,rpn_cls,roi,roi_indices,anchor = self.faster_rcnn.rpn(feature,img_size,n_pre_nms,n_post_nms,scale)

        gt_rpn_loc, gt_rpn_label = self.anchortarget(at.tonumpy(bbox), anchor, img_size)

        sample_roi, gt_roi_loc, gt_roi_label = self.propasaltarget(roi,at.tonumpy(bbox),at.tonumpy(label))

        sample_roi_index = t.zeros(len(sample_roi))#batchsize =1,全为0

        roi_cls, roi_loc = self.faster_rcnn.roihead(feature,sample_roi,sample_roi_index)

        # rpn loss
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc[0, :, :], gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_cls[0, :, :], gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_cls[0,:,:])[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        #roiloss
        n_sample = roi_loc.shape[0]
        roi_loc = roi_loc.view(n_sample, -1, 4)
        roi_loc = roi_loc[t.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)
        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, self.roi_sigma)
        roi_cls_loss = F.cross_entropy(roi_cls, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_cls, False), gt_roi_label.data.long())


        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels,n_pre_nms,n_post_nms, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels,n_pre_nms,n_post_nms,scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses


    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}



def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result



def train():

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    feat_stride = 16

    faster_rcnn = fasterrcnn(opt.n_fg_class + 1,(opt.roi_size,opt.roi_size),1./feat_stride)

    print('model construct completed')

    trainer = fasterrcnn_train(faster_rcnn).cuda()
    #读取预训练模型
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in enumerate(dataloader):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

            trainer.train_step(img, bbox, label,opt.train_n_pre_nms,opt.train_n_post_nms, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13:
            break



# def run(phase_train = True):
#     train_data = Dataset(opt)
#     dataloader = data_.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=opt.num_workers)
#     test_data = TestDataset(opt)
#     test_dataloader = data_.DataLoader(test_data,
#                                        batch_size=1,
#                                        num_workers=opt.test_num_workers,
#                                        shuffle=False,
#                                        pin_memory=True)
#     if phase_train:
#
#
#         feat_stride = 16
#
#         faster = fasterrcnn_train(opt.n_fg_class + 1,(opt.roi_size,opt.roi_size),1./feat_stride).cuda()
#
#
#         optimizer = optim.SGD(faster.parameters(),lr=opt.lr,momentum=opt.momentum,weight_decay=opt.weight_decay)
#
#         count = 0
#         scale = 1.
#         for epoch in range(opt.epoch):
#             for ii,(img,bbox_,label_,scale) in enumerate(dataloader):
#                 count += 1
#                 N,C,H,W = img.shape
#                 img_size = (H,W)
#                 scales = at.scalar(scale)
#                 img,bbox,label = img.cuda().float(),bbox_.cuda(),label_.cuda()
#
#                 losses,roi_loc_loss,roi_cls_loss,roi,rpn_loc,rpn_cls,sample_roi,gt_roi_loc,roi_loc = faster.forward(img,bbox,label,opt.train_n_pre_nms,opt.train_n_post_nms,scale)
#
#                 optimizer.zero_grad()
#                 losses.total_loss.backward()
#                 optimizer.step()
#
#                 img2 = at.tonumpy(img[0].permute(1,2,0))
#                 roitestloc = at.tonumpy(roi_loc)
#                 gtroitest = at.tonumpy(gt_roi_loc)
#                 sample = at.tonumpy(sample_roi)
#                 roitestloc = bt.loc2bbox(sample_roi,roitestloc)
#                 gtroitest = bt.loc2bbox(sample_roi,gtroitest)
#                 #c = bbox[0][0][0]
#                 #if count % 100 == 0:
#                 for i in range(10):
#                     cv2.rectangle(img2, (int(roitestloc[i][1]),int(roitestloc[i][0])), (int(roitestloc[i][3]),int(roitestloc[i][2])), (0, 255, 0), 3)
#                 #    cv2.rectangle(img2, (int(gtroitest[i][1]), int(gtroitest[i][0])),(int(gtroitest[i][3]),int(gtroitest[i][2])), (0, 0, 255), 3)
#                 cv2.imshow('image', img2)
#                 cv2.waitKey(10)
#                 if count % 10000 == 0:
#                     t.save(faster.state_dict(), 'Model/train_model.kpl')
#                     print("save model success")
#                     print(losses)
#
#
#
#             if epoch == 13:
#                 break
#     else:
#         faster = fasterrcnn_test
#
#         pred_bboxes, pred_labels,pred_scores = list(),list(),list()
#         gt_bboxes, gt_labels, gt_difficults = list(),list(),list()
#         for ii,(imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
#             sizes = [sizes[0][0]].item(),sizes[1][0].item()]
#             pred_bboxes_,pred_labels_,pred_scores_ = faster.predict(imgs, [sizes])
#
#
#
#
#
if __name__ == '__main__':
    import fire
    train()
    fire.Fire()