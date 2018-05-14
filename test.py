# -----------------------------------------------------
# Test Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Apr 10, 2018
# Latest rectified: May 11, 2018
# -----------------------------------------------------
import os
import argparse
import pickle

import numpy as np
import torch
import yaml
from torch.autograd import Variable
import time

from dataset import PersonSearchDataset
from model import SIPN
from bbox_transform import bbox_transform_inv
from nms.pth_nms import pth_nms as nms
from __init__ import clock_non_return


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--net', default='res50', type=str)
    parser.add_argument('--trained_epochs', default='10', type=str)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--out_dir', default='./output', type=str)
    parser.add_argument('--use_saved_result', default=0, type=int)
    parser.add_argument('--dataset_name', default='sysu', type=str)
    parser.add_argument('--gallery_size', default=500, type=int)

    args = parser.parse_args()

    return args


def cuda_mode(args):
    """set cuda"""
    if torch.cuda.is_available() and '-1' not in args.gpu_ids:
        cuda = True
        str_ids = args.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                gpu_ids.append(gid)

        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    else:
        cuda = False

    return cuda


def clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def test_gallery(net, dataset, use_cuda, output_dir, thresh=0.):
    """test gallery images"""

    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    num_images = len(dataset)
    all_boxes = [0 for _ in range(num_images)]
    all_features = [0 for _ in range(num_images)]
    start = time.time()

    for i in range(num_images):
        im, im_info, orig_shape = dataset.next()
        im = im.transpose([0, 3, 1, 2])

        if use_cuda:
            im = Variable(torch.from_numpy(im).cuda(), volatile=True)
        else:
            im = Variable(torch.from_numpy(im), volatile=True)

        scores, bbox_pred, rois, features = net.forward(im, None, im_info)

        boxes = rois[:, 1:5] / im_info[2]
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        if config['test_bbox_reg']:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            pred_boxes = bbox_transform_inv(
                torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
            pred_boxes = clip_boxes(pred_boxes, orig_shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        boxes = pred_boxes

        # skip j = 0, because it's the background class
        j = 1
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = nms(torch.from_numpy(cls_dets),
                   config['test_nms']).numpy() if cls_dets.size > 0 else []
        cls_dets = cls_dets[keep, :]
        all_boxes[i] = cls_dets
        all_features[i] = features[inds][keep]

        end = time.time()
        print('im_detect: {:d}/{:d} {:.3f}s'.format(
            i + 1, num_images, (end - start) / (i + 1)))

    det_file = os.path.join(output_dir, 'gboxes.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    feature_file = os.path.join(output_dir, 'gfeatures.pkl')
    with open(feature_file, 'wb') as f:
        pickle.dump(all_features, f, pickle.HIGHEST_PROTOCOL)

    return all_boxes, all_features


def test_query(net, dataset, use_cuda, output_dir):
    """Test query images"""

    # TODO: use __len__()
    num_images = dataset.queries_to_galleries.shape[0]
    all_features = [0 for _ in range(num_images)]
    start = time.time()

    for i in range(num_images):
        im, roi, im_info = dataset.next()

        im = im.transpose([0, 3, 1, 2])
        roi = np.hstack(([[0]], roi.reshape(1, 4)))

        if use_cuda:
            im = Variable(torch.from_numpy(im).cuda(), volatile=True)
            roi = Variable(torch.from_numpy(roi).float().cuda())
        else:
            im = Variable(torch.from_numpy(im), volatile=True)
            roi = Variable(torch.from_numpy(roi).float())

        features = net.forward(im, roi, im_info, dataset.test_mode)
        all_features[i] = features[0]  # TODO: check this

        end = time.time()
        print('query_exfeat: {:d}/{:d} {:.3f}s'.format(
            i + 1, num_images, (end - start) / (i + 1)))

    feature_file = os.path.join(output_dir, 'qfeatures.pkl')
    with open(feature_file, 'wb') as f:
        pickle.dump(all_features, f, pickle.HIGHEST_PROTOCOL)

    return all_features


@clock_non_return
def main():
    """Test the model"""

    opt = parse_args()
    size = opt.gallery_size
    test_result_dir = os.path.join(opt.out_dir, opt.dataset_name,
                                   'test_result')

    dataset_gallery = PersonSearchDataset(opt.data_dir, opt.dataset_name,
                                          split_name='test', gallery_size=size)

    if opt.use_saved_result:
        gboxes_file = open(os.path.join(test_result_dir, 'gboxes.pkl'), 'rb')
        g_boxes = pickle.load(gboxes_file)
        gfeatures_file = open(os.path.join(test_result_dir,
                                           'gfeatures.pkl'), 'rb')
        g_features = pickle.load(gfeatures_file)
        qfeatures_file = open(os.path.join(test_result_dir, 'qfeatures.pkl'),
                              'rb')
        q_features = pickle.load(qfeatures_file)

    else:
        use_cuda = cuda_mode(opt)

        trained_model_dir = os.path.join(
            opt.out_dir, opt.dataset_name, 'sipn_' + opt.net + '_' +
                                           opt.trained_epochs + '.pth')

        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)

        net = SIPN(opt.net, opt.dataset_name, trained_model_dir,
                   is_train=False)
        net.eval()
        if use_cuda:
            net.cuda()

        # load trained model
        print('Loading model check point from {:s}'.format(trained_model_dir))
        net.load_trained_model(torch.load(trained_model_dir))

        dataset_query = PersonSearchDataset(
            opt.data_dir, opt.dataset_name, split_name='test',
            gallery_size=size, test_mode='query')

        g_boxes, g_features = test_gallery(net, dataset_gallery, use_cuda,
                                            test_result_dir)
        q_features = test_query(net, dataset_query, use_cuda, test_result_dir)

    dataset_gallery.evaluate_detections(g_boxes, det_thresh=0.5)
    dataset_gallery.evaluate_search(g_boxes, g_features, q_features,
                                    det_thresh=0.5, gallery_size=size,
                                    dump_json=None)


if __name__ == '__main__':

    main()
