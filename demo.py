# -----------------------------------------------------
# Demo Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Apr 26, 2018
# Latest rectified: Oct 25, 2018
# -----------------------------------------------------
import os
import argparse

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import clock_non_return
from dataset.sipn_dataset import pre_process_image
from models.model import SIPN
from utils.bbox_transform import bbox_transform_inv
from nms.pth_nms import pth_nms as nms


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--net', default='res50', type=str)
    parser.add_argument('--trained_epochs', default='10', type=str)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='./demo', type=str)
    parser.add_argument('--model_dir', default='./output', type=str)
    parser.add_argument('--dataset_name', default='sysu', type=str)

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


def demo_detection(net, im_dir, images, use_cuda, thresh=.75):
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    with torch.no_grad():
        for im_name in images:
            im_path = os.path.join(im_dir, im_name)
            im, im_scale, orig_shape = pre_process_image(im_path, copy=True)
            im_info = np.array([im.shape[1], im.shape[2], im_scale],
                               dtype=np.float32)

            im = im.transpose([0, 3, 1, 2])

            if use_cuda:
                im = torch.from_numpy(im).cuda()
            else:
                im = torch.from_numpy(im)

            scores, bbox_pred, rois, _ = net.forward(im, None, im_info)

            boxes = rois[:, 1:5] / im_info[2]
            scores = np.reshape(scores, [scores.shape[0], -1])
            bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
            if config['test_bbox_reg']:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred
                pred_boxes = bbox_transform_inv(
                    torch.from_numpy(boxes),
                    torch.from_numpy(box_deltas)).numpy()
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
            cls_dets = np.hstack(
                (cls_boxes, cls_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(torch.from_numpy(cls_dets),
                       config['test_nms']).numpy() if cls_dets.size > 0 else []
            cls_dets = cls_dets[keep, :]

            if cls_dets is None:
                print('There are no detections in image {}'.format(im_name))
                continue

            fig, ax = plt.subplots(figsize=(16, 9))
            ax.imshow(plt.imread(im_path))
            plt.axis('off')
            for box in cls_dets:
                x1, y1, x2, y2, score = box
                ax.add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False,
                    edgecolor='#66D9EF', linewidth=3.5))
                ax.add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False,
                    edgecolor='white', linewidth=1))
                ax.text(x1 + 5, y1 - 15, '{:.2f}'.format(score),
                        bbox=dict(facecolor='#66D9EF', linewidth=0),
                        fontsize=20, color='white')
            plt.tight_layout()
            plt.show()
            plt.close(fig)


def demo_search(net, im_dir, images, use_cuda, thresh=.75):
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    q_name = 's15166.jpg'
    q_roi = [29, 5, 164, 439]  # x1, y1, h, w
    x1, y1, h, w = q_roi

    q_path = os.path.join(im_dir, q_name)
    q_im, q_scale, _ = pre_process_image(q_path)
    q_roi = np.array(q_roi) * q_scale
    q_info = np.array([q_im.shape[1], q_im.shape[2], q_scale],
                      dtype=np.float32)

    q_im = q_im.transpose([0, 3, 1, 2])
    q_roi = np.hstack(([[0]], q_roi.reshape(1, 4)))

    with torch.no_grad():
        if use_cuda:
            q_im = torch.from_numpy(q_im).cuda()
            q_roi = torch.from_numpy(q_roi).float().cuda()
        else:
            q_im = torch.from_numpy(q_im)
            q_roi = torch.from_numpy(q_roi).float()

        q_feat = net.forward(q_im, q_roi, q_info, 'query')[0]

    # Show query
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(plt.imread(q_path))
    plt.axis('off')
    ax.add_patch(plt.Rectangle((x1, y1), h, w, fill=False, edgecolor='#F92672',
                               linewidth=3.5))
    ax.add_patch(plt.Rectangle((x1, y1), h, w, fill=False, edgecolor='white',
                               linewidth=1))
    ax.text(x1 + 5, y1 - 15, '{}'.format('Query'),
            bbox=dict(facecolor='#F92672', linewidth=0), fontsize=20,
            color='white')
    plt.tight_layout()
    fig.savefig(os.path.join(im_dir, 'query.jpg'))
    plt.show()
    plt.close(fig)

    # Get gallery images
    images.remove(q_name)
    for im_name in images:
        im_path = os.path.join(im_dir, im_name)
        im, im_scale, orig_shape = pre_process_image(im_path, copy=True)
        im_info = np.array([im.shape[1], im.shape[2], im_scale],
                           dtype=np.float32)

        im = im.transpose([0, 3, 1, 2])

        if use_cuda:
            im = torch.from_numpy(im).cuda()
        else:
            im = torch.from_numpy(im)

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
        features = features[inds][keep]

        if cls_dets is None:
            print('There are no detections in image {}'.format(im_name))
            continue

        similarities = features.dot(q_feat)

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(plt.imread(im_path))
        plt.axis('off')

        # Set different colors for different ids
        similarities_list = similarities.tolist()
        max_sim = max(similarities_list)
        similarities_list.remove(max_sim)
        colors = {value: '#66D9EF' for value in similarities_list}
        colors[max_sim] = '#4CAF50'

        for box, sim in zip(cls_dets, similarities):
            x1, y1, x2, y2, _ = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=colors[sim], linewidth=3.5))
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor='white', linewidth=1))
            ax.text(x1 + 5, y1 - 15, '{:.2f}'.format(sim),
                    bbox=dict(facecolor=colors[sim], linewidth=0), fontsize=20,
                    color='white')
        plt.tight_layout()
        fig.savefig(os.path.join(im_dir, 'result_' + im_name))
        plt.show()
        plt.close(fig)


@clock_non_return
def main():

    opt = parse_args()
    use_cuda = cuda_mode(opt)

    trained_model_dir = os.path.join(
        opt.model_dir, opt.dataset_name, 'sipn_' + opt.net + '_' +
                                         opt.trained_epochs + '.pth')

    net = SIPN(opt.net, opt.dataset_name, trained_model_dir, is_train=False)
    net.eval()
    if use_cuda:
        net.cuda()

    # load trained model
    print('Loading model check point from {:s}'.format(trained_model_dir))
    net.load_trained_model(torch.load(trained_model_dir))

    test_images = os.listdir(opt.data_dir)

    # demo_detection(net, opt.data_dir, test_images, use_cuda)
    demo_search(net, opt.data_dir, test_images, use_cuda)


if __name__ == '__main__':

    main()
