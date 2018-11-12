# -----------------------------------------------------
# Person Search Dataset for Both Training and Testing
#
# Author: Liangqi Li
# Creating Date: Mar 28, 2018
# Latest rectified: Nov 5, 2018
# -----------------------------------------------------
import os
import os.path as osp
import random
from itertools import permutations

import pandas as pd
import cv2
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from sklearn.metrics import average_precision_score, precision_recall_curve


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
            (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def sipn_fn(batch):

    assert len(batch[0]) == 3 or len(batch[0]) == 4
    im_tensors = torch.stack([x[0] for x in batch])
    gt_boxes = torch.stack([x[1] for x in batch])
    im_info = np.vstack([x[2] for x in batch])

    return im_tensors, gt_boxes, im_info


class SIPNQueryDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(self.root_dir, 'frames')
        self.anno_dir = os.path.join(self.root_dir, 'SIPN_annotation')
        self.query_boxes = pd.read_csv(os.path.join(
            self.anno_dir, 'queryDF.csv'))

    def __len__(self):
        return self.query_boxes.shape[0]

    def __getitem__(self, index):
        im_name = self.query_boxes.loc[index, 'imname']
        im_path = os.path.join(self.image_dir, im_name)
        box = self.query_boxes.loc[index, 'x1': 'del_y'].copy()
        box['del_x'] += box['x1']
        box['del_y'] += box['y1']
        box = box.values.astype(np.float32)

        im = cv2.imread(im_path).astype(np.float32)
        if self.transform is not None:
            im, im_scale, _ = self.transform(im)
        else:
            im_scale = 1
        im_info = np.array(
            [im.shape[1], im.shape[2], im_scale], dtype=np.float32)

        box *= im_scale
        box = torch.Tensor(box).float()

        return im, (box, im_info)


class SIPNDataset(Dataset):

    def __init__(self, root_dir, dataset_name, split, transform=None):
        """
        Create the SIPN dataset
        ---
        :param root_dir: (str) root path for the raw dataset
        :param split: (str) 'train' or 'test'
        """

        self.root_dir = root_dir
        self.split = split
        self.dataset_name = dataset_name
        self.transform = transform

        self.image_dir = os.path.join(self.root_dir, 'frames')
        self.anno_dir = os.path.join(self.root_dir, 'SIPN_annotation')
        self.imnames_file = '{}ImnamesSe.csv'.format(self.split)
        self.all_boxes_file = '{}AllDF.csv'.format(self.split)

        self.all_boxes = pd.read_csv(os.path.join(
            self.anno_dir, self.all_boxes_file))
        self.imnames = pd.read_csv(os.path.join(
                self.anno_dir, self.imnames_file), header=None, squeeze=True)

        # Count all the pids except for -1
        self.pids = list(set(self.all_boxes['pid']) - {-1})
        random.shuffle(self.pids)

        # TODO: remove the below lines
        if self.dataset_name == 'sysu':
            self.gallery_sizes = [50, 100, 500, 1000, 2000, 4000]
            self.num_pid = 5532
        elif self.dataset_name == 'prw':
            self.gallery_sizes = [200, 500, 1000, 2000, 4000]
            self.num_pid = 483
        else:
            raise KeyError(self.dataset_name)

    def __len__(self):
        return self.imnames.shape[0]

    def __getitem__(self, index):
        im_name = self.imnames[index]
        im_path = os.path.join(self.image_dir, im_name)
        boxes_df = self.all_boxes.query('imname==@im_name')
        boxes = boxes_df.loc[:, 'x1': 'pid'].copy()
        boxes.loc[:, 'del_x'] += boxes.loc[:, 'x1']
        boxes.loc[:, 'del_y'] += boxes.loc[:, 'y1']
        boxes = boxes.values.astype(np.float32)

        im = cv2.imread(im_path).astype(np.float32)
        orig_shape = im.shape
        if self.transform is not None:
            im, im_scale, flip = self.transform(im)
        else:
            im_scale = 1
            flip = False
        im_info = np.array(
            [im.shape[1], im.shape[2], im_scale], dtype=np.float32)

        width = orig_shape[1]
        boxes_temp = boxes.copy()
        if flip:
            boxes[:, 2] = width - boxes_temp[:, 0] - 1
            boxes[:, 0] = width - boxes_temp[:, 2] - 1

        boxes[:, :4] *= im_scale
        boxes = torch.Tensor(boxes).float()

        if self.split == 'train':
            return im, boxes, im_info, im_name
        elif self.split == 'test':
            return im, (orig_shape, im_info)
        else:
            raise KeyError(self.split)

    def evaluate_detections(self, gallery_det, det_thresh=0.5, iou_thresh=0.5,
                            labeled_only=False):
        """evaluate the results of the detection"""
        assert self.imnames.shape[0] == len(gallery_det)

        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        df = self.all_boxes.copy()
        for k in range(len(gallery_det)):
            im_name = self.imnames.iloc[k]
            gt = df[df['imname'] == im_name]
            gt_boxes = gt.loc[:, 'x1': 'del_y'].copy()
            gt_boxes.loc[:, 'del_x'] += gt_boxes.loc[:, 'x1']
            gt_boxes.loc[:, 'del_y'] += gt_boxes.loc[:, 'y1']
            gt_boxes = gt_boxes.values

            if labeled_only:
                pass  # TODO
            det = np.asarray(gallery_det[k])
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= iou_thresh)

            # for each det, keep only the largest iou of all the gt
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in range(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in range(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in range(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False

            for j in range(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        precision, recall, __ = precision_recall_curve(y_true, y_score)
        recall *= det_rate

        # plt.plot(recall, precision)
        # plt.savefig('pr.jpg')

        print('Detection results:')
        print('  Recall = {:.2%}'.format(det_rate))
        if not labeled_only:
            print('  AP = {:.2%}'.format(ap))
        print('*' * 20)
        print()

    def evaluate_search(self, gallery_det, gallery_feat, probe_feat,
                        det_thresh=0.5, gallery_size=100):
        assert self.imnames.shape[0] == len(gallery_det)
        assert self.imnames.shape[0] == len(gallery_feat)

        query_boxes = pd.read_csv(osp.join(self.anno_dir, 'queryDF.csv'))
        q_to_g_file = 'q_to_g' + str(gallery_size) + 'DF.csv'
        queries_to_galleries = pd.read_csv(osp.join(
            self.anno_dir, q_to_g_file))
        assert queries_to_galleries.shape[0] == len(probe_feat)

        use_full_set = gallery_size == -1
        df = self.all_boxes.copy()

        # ====================formal=====================
        name_to_det_feat = {}
        for name, det, feat in zip(self.imnames, gallery_det, gallery_feat):
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

        # # =====================debug=====================
        # f = open('name_to_det_feat.pkl', 'rb+')
        # name_to_det_feat = pickle.load(f)
        # # ======================end======================

        aps = []
        accs = []
        topk = [1, 5, 10]
        for i in range(len(probe_feat)):
            pid = int(query_boxes.ix[i, 'pid'])
            assert isinstance(pid, int)
            num_g = query_boxes.ix[i, 'num_g']
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()
            # Ignore the probe image
            probe_imname = queries_to_galleries.iloc[i, 0]
            probe_gt = []
            tested = {probe_imname}
            # 1. Go through the gallery samples defined by the protocol
            for g_i in range(1, gallery_size + 1):
                gallery_imname = queries_to_galleries.iloc[i, g_i]
                # gt = df[df['imname'] == gallery_imname]
                # gt = gt[gt['pid'] == pid]  # important
                # gt = gt.loc[:, 'x1': 'y2']
                if g_i <= num_g:
                    gt = df.query('imname==@gallery_imname and pid==@pid')
                    gt = gt.loc[:, 'x1': 'del_y'].copy()
                    gt.loc[:, 'del_x'] += gt.loc[:, 'x1']
                    gt.loc[:, 'del_y'] += gt.loc[:, 'y1']
                    gt = gt.values.ravel()
                else:
                    gt = np.array([])
                count_gt += (gt.size > 0)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat:
                    continue
                det, feat_g = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:
                    w, h = gt[2] - gt[0], gt[3] - gt[1]
                    probe_gt.append({'img': str(gallery_imname),
                                     'roi': list(gt.astype('float'))})
                    iou_thresh = min(.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if _compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)
            # 2. Go through the remaining gallery images if using full set
            if use_full_set:
                pass  # TODO
            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            if count_gt == 0:
                print(probe_imname, i)
                break
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else average_precision_score(
                y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            # y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])

            if (i + 1) % 100 == 0:
                print('Evaluating the {}-th query'.format(i+1))

        print()
        print('Search ranking:')
        print('  mAP = {:.2%}'.format(np.mean(aps)))
        accs = np.mean(accs, axis=0)
        for i, k in enumerate(topk):
            print('  top-{:2d} = {:.2%}'.format(k, accs[i]))


class PersonSearchTripletSampler(Sampler):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        all_boxes_df = dataset.all_boxes
        all_imnames = dataset.imnames
        all_pids = dataset.pids

        print('Sampling the identities to triplets...')
        iter_inds_path = os.path.join(dataset.anno_dir, 'tri_iter_inds.pkl')
        batch_pids_path = os.path.join(dataset.anno_dir, 'tri_batch_pids.pkl')
        if os.path.exists(iter_inds_path) and os.path.exists(batch_pids_path):
            with open(iter_inds_path, 'rb') as f:
                self.iter_inds = pickle.load(f)
            with open(batch_pids_path, 'rb') as f:
                self.batch_pids = pickle.load(f)

        else:
            self.iter_inds = []
            self.batch_pids = []

            # For each pid, select an anchor image and a positive image
            for i, pid in enumerate(all_pids):
                if (i+1) % 100 == 0:
                    print('Processing identity {}/{}'.format(
                        i+1, len(all_pids)))
                cur_pid_df = all_boxes_df.query('pid==@pid')
                pos_imnames = list(set(cur_pid_df['imname']))
                assert len(pos_imnames) >= 2
                # Select negative images from images that don't contain the pid
                neg_imnames = list(set(all_imnames) - set(pos_imnames))

                # Get An2 permutations for the anchor and the positive
                for q_name, p_name in permutations(pos_imnames, 2):
                    names = [q_name, p_name, random.choice(neg_imnames)]
                    # Get the index of the very image in dataset.imnames
                    ids = [all_imnames[all_imnames == name].index.tolist()[0]
                           for name in names]
                    self.iter_inds.append(ids)
                    self.batch_pids.append(pid)

            with open(iter_inds_path, 'wb') as f:
                pickle.dump(self.iter_inds, f, pickle.HIGHEST_PROTOCOL)
            with open(batch_pids_path, 'wb') as f:
                pickle.dump(self.batch_pids, f, pickle.HIGHEST_PROTOCOL)

        print('Done.\n')

    def __iter__(self):
        for inds in self.iter_inds:
            yield inds

    def __len__(self):
        return len(self.dataset)


class PersonSearchTripletFn:

    def __init__(self, dataset, batch_pids):
        self.dataset = dataset
        self.batch_pids = batch_pids
        self.called_times = 0

    def __call__(self, batch):
        assert len(batch) == 3
        assert len(batch[0]) == 4
        # Transfer tuple to list to change its value then
        for i in range(3):
            batch[i] = [x for x in batch[i]]

        # Pick out the DataFrame of the current image and current identity
        pid = self.batch_pids[self.called_times]
        q_name = batch[0][-1]
        assert isinstance(q_name, str)
        q_boxes = self.dataset.all_boxes.query('imname==@q_name')
        q_box = self.dataset.all_boxes.query('imname==@q_name and pid==@pid')
        assert q_box.shape[0] == 1

        # Get the index where the identity appears in the current image
        q_boxes = q_boxes.loc[:, 'x1': 'del_y'].values
        q_box = q_box.loc[:, 'x1': 'del_y'].values.ravel()
        idx = np.all(q_boxes == q_box, axis=1).nonzero()
        assert len(idx) == 1

        # Change the boxes of the image to the box of the current identity
        idx = idx[0].item()
        self.called_times += 1
        batch[0][1] = batch[0][1][idx].unsqueeze(0)
        assert int(batch[0][1][0, -1].item()) == pid

        # Crop the query image to a query person
        q_box = batch[0][1][0].numpy()[:4].astype(np.int32).tolist()
        x1, y1, x2, y2 = q_box
        assert y2 <= batch[0][0].size(1) and x2 <= batch[0][0].size(2), \
            '{} with size ({}, {}) conflicts with box {}'.format(
                q_name, batch[0][0].size(2), batch[0][0].size(1), q_box)
        assert x1 < x2 and y1 < y2, '{} has conflict box {}'.format(
            q_name, q_box)
        batch[0][0] = batch[0][0][:, y1: y2, x1: x2]

        im_tensors = tuple([x[0].unsqueeze(0) for x in batch])
        gt_boxes = tuple([x[1] for x in batch])
        im_info = tuple([x[2] for x in batch])

        return im_tensors, gt_boxes, im_info


if __name__ == '__main__':

    # rtdir = '/home/liliangqi/hdd/datasets/cuhk_sysu'
    #
    # cur_transform = sipn_transforms.Compose([
    #     sipn_transforms.Scale(600, 1000),
    #     sipn_transforms.ToTensor(),
    #     sipn_transforms.Normalize([102.9801, 115.9465, 122.7717])
    # ])
    # gallery_dataset = SIPNDataset(rtdir, 'sysu', 'test', cur_transform)

    print('Debug')
