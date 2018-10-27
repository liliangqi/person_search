# -----------------------------------------------------
# Test Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Apr 10, 2018
# Latest rectified: Oct 27, 2018
# -----------------------------------------------------
import os
import time

import argparse
import yaml
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.sipn_dataset import SIPNQueryDataset, SIPNDataset
import dataset.sipn_transforms as sipn_transforms
from models.model import SIPN
from utils.bbox_transform import bbox_transform_inv
from nms.pth_nms import pth_nms as nms
from utils.utils import clock_non_return, clip_boxes, AverageMeter


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--net', default='res50', type=str)
    parser.add_argument('--epochs', default='20', type=str)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--out_dir', default='./output', type=str)
    parser.add_argument('--use_saved_result', default=1, type=int)
    parser.add_argument('--dataset_name', default='prw', type=str)
    parser.add_argument('--gallery_size', default=200, type=int)

    args = parser.parse_args()

    return args


def test_gallery(net, dataloader, output_dir, thresh=0.):
    """test gallery images"""

    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    num_images = len(dataloader.dataset)
    all_boxes = []
    all_features = []
    end = time.time()
    time_cost = AverageMeter()
    net.eval()

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            im, (orig_shape, im_info) = data
            im = im.to(device)
            im_info = im_info.numpy().squeeze(0)
            orig_shape = [x.item() for x in orig_shape]

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
        all_boxes.append(cls_dets)
        all_features.append(features[inds][keep])

        time_cost.update(time.time() - end)
        end = time.time()
        print('im_detect: {:d}/{:d} {:.3f}s'.format(
            i + 1, num_images, time_cost.avg))

    det_file = os.path.join(output_dir, 'gboxes.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    feature_file = os.path.join(output_dir, 'gfeatures.pkl')
    with open(feature_file, 'wb') as f:
        pickle.dump(all_features, f, pickle.HIGHEST_PROTOCOL)

    return all_boxes, all_features


def test_query(net, dataloader, output_dir):
    """Test query images"""

    num_images = len(dataloader.dataset)
    all_features = []
    end = time.time()
    time_cost = AverageMeter()
    net.eval()

    for i, data in enumerate(dataloader):
        im, (roi, im_info) = data
        im = im.to(device)
        roi = torch.cat((torch.zeros(1, 1), roi), 1).to(device)

        with torch.no_grad():
            features = net.forward(im, roi, im_info, 'query')
        all_features.append(features[0])  # TODO: check this

        time_cost.update(time.time() - end)
        end = time.time()
        print('query_exfeat: {:d}/{:d} {:.3f}s'.format(
            i + 1, num_images, time_cost.avg))

    feature_file = os.path.join(output_dir, 'qfeatures.pkl')
    with open(feature_file, 'wb') as f:
        pickle.dump(all_features, f, pickle.HIGHEST_PROTOCOL)

    return all_features


@clock_non_return
def main():
    """Test the model"""

    opt = parse_args()
    size = opt.gallery_size
    test_result_dir = os.path.join(
        opt.out_dir, opt.dataset_name, 'test_result')

    # Read the configuration file
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    target_size = config['target_size']
    max_size = config['max_size']
    pixel_means = config['pixel_means']

    # Compose transformations for the dataset
    transform = sipn_transforms.Compose([
        sipn_transforms.Scale(target_size, max_size),
        sipn_transforms.ToTensor(),
        sipn_transforms.Normalize(pixel_means)
    ])

    dataset_gallery = SIPNDataset(
        opt.data_dir, opt.dataset_name, 'test', transform)

    if opt.use_saved_result:
        with open(os.path.join(test_result_dir, 'gboxes.pkl'), 'rb') as f:
            g_boxes = pickle.load(f)
        with open(os.path.join(test_result_dir, 'gfeatures.pkl'), 'rb') as f:
            g_features = pickle.load(f)
        with open(os.path.join(test_result_dir, 'qfeatures.pkl'), 'rb') as f:
            q_features = pickle.load(f)

    else:
        global device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net = SIPN(opt.net, opt.dataset_name)
        net.to(device)

        # Load trained model
        trained_model_name = 'sipn_{}_{}.tar'.format(opt.net, opt.epochs)
        trained_model_dir = os.path.join(
            opt.out_dir, opt.dataset_name, trained_model_name)
        print('Loading model check point from {:s}'.format(trained_model_dir))
        checkpoint = torch.load(trained_model_dir)
        net.load_trained_model(checkpoint['model_state_dict'])

        # Define datasets
        dataset_query = SIPNQueryDataset(opt.data_dir, transform)
        query_loader = DataLoader(dataset_query, num_workers=8)
        gallery_loader = DataLoader(dataset_gallery, num_workers=8)

        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        q_features = test_query(net, query_loader, test_result_dir)
        g_boxes, g_features = test_gallery(
            net, gallery_loader, test_result_dir)

    dataset_gallery.evaluate_detections(g_boxes, det_thresh=0.5)
    dataset_gallery.evaluate_search(g_boxes, g_features, q_features,
                                    det_thresh=0.5, gallery_size=size)


if __name__ == '__main__':
    main()
