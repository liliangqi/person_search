# -----------------------------------------------------
# Person Search Dataset for Both Training and Testing
#
# Author: Liangqi Li
# Creating Date: Mar 28, 2018
# Latest rectified: Apr 28, 2018
# -----------------------------------------------------

import os
import os.path as osp
import time

import yaml
import pandas as pd
import cv2
import PIL
import random
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import pickle
import matplotlib.pyplot as plt


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
            (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def pre_process_image(im_path, flipped=0, copy=False):
    """Pre-process the image"""

    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    target_size = config['target_size']
    max_size = config['max_size']
    pixel_means = np.array([[config['pixel_means']]])

    im = cv2.imread(im_path)
    orig_shape = im.shape
    if flipped == 1:
        im = im[:, ::-1, :]
    im = im.astype(np.float32, copy=copy)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im = im[np.newaxis, :]  # add batch dimension

    return im, im_scale, orig_shape


class PersonSearchDataset:

    def __init__(self, root_dir, split_name='train', gallery_size=100,
                 test_mode='gallery'):
        """
        create person search dataset
        ---
        param:
            root_dir: root direction for the raw dataset
            split_name: 'train' or 'test'
        """
        self.root_dir = root_dir
        self.split = split_name
        self.images_dir = osp.join(self.root_dir, 'Image/SSM')
        self.annotation_dir = osp.join(self.root_dir, 'annotation')
        self.cache_dir = osp.join(self.root_dir, '..', 'cache')

        self.train_imnames_file = 'trainImnamesSe.csv'
        self.train_imnamesDF_file = 'trainImnamesDF.csv'
        self.test_imnames_file = 'testImnamesSe.csv'
        self.train_all_file = 'trainAllDF.csv'
        self.test_all_file = 'testAllDF.csv'
        self.query_file = 'queryDF.csv'
        self.gallery_sizes = [50, 100, 500, 1000, 2000, 4000]

        if not osp.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        if self.split == 'train':
            if self.train_imnamesDF_file in os.listdir(self.cache_dir) and \
                    self.train_all_file in os.listdir(self.cache_dir):
                self.train_imnames = pd.read_csv(
                    osp.join(self.cache_dir, self.train_imnamesDF_file))
                self.train_all = pd.read_csv(
                    osp.join(self.cache_dir, self.train_all_file))
            else:
                self.train_imnames, self.train_all = self.prepare_training()
            self.train_imnames_list = list(range(self.train_imnames.shape[0]))
            random.shuffle(self.train_imnames_list)  # shuffle the list
            self.train_imnames_list_equip = self.train_imnames_list[:]
            self.num_train_images = self.train_imnames.shape[0]

        elif self.split == 'test':
            self.test_imnames = pd.read_csv(
                osp.join(self.annotation_dir, self.test_imnames_file),
                header=None, squeeze=True)
            self.test_all = pd.read_csv(
                osp.join(self.annotation_dir, self.test_all_file))
            q_to_g_file = 'q_to_g' + str(gallery_size) + 'DF.csv'
            self.queries_to_galleries = pd.read_csv(
                osp.join(self.annotation_dir, q_to_g_file))
            self.query_boxes = pd.read_csv(osp.join(
                self.annotation_dir, self.query_file))
            self.delta_to_coordinates()
            self.num_test_images = self.test_imnames.shape[0]
            self.test_imnames_list = list(range(self.num_test_images))[::-1]
            # random.shuffle(self.test_imnames_list)

            # TODO: split query and gallery completely
            self.test_mode = test_mode
            if self.test_mode == 'query':
                self.query_imnames_list = list(range(
                    self.queries_to_galleries.shape[0]))[::-1]

        else:
            raise KeyError(self.split)

    def delta_to_coordinates(self):
        """change `del_x` and `del_y` to `x2` and `y2` for testing set"""
        self.test_all['del_x'] += self.test_all['x1']
        self.test_all['del_y'] += self.test_all['y1']
        self.test_all.rename(columns={'del_x': 'x2', 'del_y': 'y2'},
                             inplace=True)
        self.query_boxes['del_x'] += self.query_boxes['x1']
        self.query_boxes['del_y'] += self.query_boxes['y1']
        self.query_boxes.rename(columns={'del_x': 'x2', 'del_y': 'y2'},
                                inplace=True)

    def prepare_training(self):
        """prepare dataset for training, including flipping, resizing"""

        def flip_bbox(subset, train_al, hei_and_wid):
            """flip bboxes only"""
            subset = pd.merge(subset, hei_and_wid)
            subset['x1'] = subset['width'] - train_al['x2'] - 1
            subset['x2'] = subset['width'] - train_al['x1'] - 1
            assert (subset['x2'] >= subset['x1']).all()
            del subset['width']
            del subset['height']
            subset['flipped'] = [1 for _ in range(subset.shape[0])]

            return subset

        train_imnames = pd.read_csv(osp.join(self.annotation_dir,
                                             self.train_imnames_file),
                                    header=None, squeeze=True)
        train_all = pd.read_csv(osp.join(self.annotation_dir,
                                         self.train_all_file))

        # formal
        widths = [PIL.Image.open(osp.join(self.images_dir, imname)).size[0]
                  for imname in train_imnames]
        heights = [PIL.Image.open(osp.join(self.images_dir, imname)).size[1]
                   for imname in train_imnames]
        heights_and_widths = pd.DataFrame({'imname': train_imnames,
                                           'width': widths,
                                           'height': heights})

        # # ==========================debug============================
        # heights_and_widths = pd.read_csv(
        #     osp.join(self.annotation_dir, 'heights_and_widthsDF.csv'))
        # # ===========================end=============================

        # change `del_x` and `del_y` to `x2` and `y2`
        train_all['del_x'] += train_all['x1']
        train_all['del_y'] += train_all['y1']
        train_all.rename(columns={'del_x': 'x2', 'del_y': 'y2'}, inplace=True)

        # horizontally flip bounding boxes
        flipped = [0 for _ in range(train_all.shape[0])]
        train_all['flipped'] = flipped
        train_add = train_all.copy()
        train_add = flip_bbox(train_add, train_all, heights_and_widths)
        train_all = pd.concat((train_all, train_add))
        train_all.index = range(train_all.shape[0])

        flipped = [0 for _ in range(train_imnames.size)]
        train_imnames_not = pd.DataFrame({'imname': train_imnames,
                                          'flipped': flipped})
        # change the order of columns
        train_imnames_not = train_imnames_not[['imname', 'flipped']]
        flipped = [1 for _ in range(train_imnames.size)]
        train_imnames_fl = pd.DataFrame({'imname': train_imnames,
                                         'flipped': flipped})
        train_imnames_fl = train_imnames_fl[['imname', 'flipped']]
        train_imnames = pd.concat((train_imnames_fl, train_imnames_not))
        train_imnames.index = range(train_imnames.shape[0])

        train_imnames.to_csv(osp.join(self.cache_dir, 'trainImnamesDF.csv'),
                             index=False)
        train_all.to_csv(osp.join(self.cache_dir, 'trainAllDF.csv'),
                         index=False)

        return train_imnames, train_all

    def next(self):
        if self.split == 'train':

            # Prepare for the next epoch
            if len(self.train_imnames_list) == 0:
                self.train_imnames_list = self.train_imnames_list_equip[:]
                # self.train_imnames = pd.read_csv(
                #     osp.join(self.cache_dir, self.train_imnamesDF_file))

            chosen = self.train_imnames_list.pop()
            im_name, flipped = self.train_imnames.loc[chosen]
            im_path = osp.join(self.images_dir, im_name)

            im, im_scale, _ = pre_process_image(im_path, flipped=flipped)

            df = self.train_all.copy()
            # TODO: use panda.query
            df = df[df['imname'] == im_name]
            df = df[df['flipped'] == flipped]
            gt_boxes = df.loc[:, 'x1': 'pid']
            gt_boxes.loc[:, 'x1': 'y2'] *= im_scale
            gt_boxes = gt_boxes.values

            im_info = np.array([im.shape[1], im.shape[2], im_scale],
                               dtype=np.float32)

            return im, gt_boxes, im_info

        elif self.split == 'test':
            if self.test_mode == 'gallery':
                chosen = self.test_imnames_list.pop()
                im_name = self.test_imnames.loc[chosen]
                im_path = osp.join(self.images_dir, im_name)

                im, im_scale, orig_shape = pre_process_image(im_path,
                                                             copy=True)

                im_info = np.array([im.shape[1], im.shape[2], im_scale],
                                   dtype=np.float32)

                return im, im_info, orig_shape, im_path

            elif self.test_mode == 'query':
                chosen = self.query_imnames_list.pop()
                im_name = self.query_boxes.iloc[chosen, 0]
                im_path = osp.join(self.images_dir, im_name)

                im, im_scale, _ = pre_process_image(im_path, copy=True)

                df = self.query_boxes.copy()
                gt_boxes = df.ix[chosen, 'x1': 'y2'] * im_scale
                gt_boxes = gt_boxes.as_matrix().astype(np.float64)

                im_info = np.array([im.shape[1], im.shape[2], im_scale],
                                   dtype=np.float32)

                return im, gt_boxes, im_info

            else:
                raise KeyError(self.test_mode)

        else:
            raise KeyError(self.split)

    def test_image_path(self, i):
        return osp.join(self.images_dir, self.test_imnames.iloc[i])

    def __len__(self):
        if self.split == 'train':
            return self.num_train_images
        elif self.split == 'test':
            return self.num_test_images
        else:
            raise KeyError(self.split)

    def evaluate_detections(self, gallery_det, det_thresh=0.5, iou_thresh=0.5,
                            labeled_only=False):
        """evaluate the results of the detection"""
        assert self.test_imnames.shape[0] == len(gallery_det)

        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        df = self.test_all.copy()
        for k in range(len(gallery_det)):
            im_name = self.test_imnames.iloc[k]
            gt = df[df['imname'] == im_name]
            gt_boxes = gt.loc[:, 'x1': 'y2'].values
            if labeled_only:
                pass # TODO
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

        plt.plot(recall, precision)
        plt.savefig('pr.jpg')

        print(
            '{} detection:'.format('labeled only' if labeled_only else 'all'))
        print('  recall = {:.2%}'.format(det_rate))
        if not labeled_only:
            print('  ap = {:.2%}'.format(ap))

    def evaluate_search(self, gallery_det, gallery_feat, probe_feat,
                        det_thresh=0.5, gallery_size=100, dump_json=None):
        assert self.test_imnames.shape[0] == len(gallery_det)
        assert self.test_imnames.shape[0] == len(gallery_feat)
        assert self.queries_to_galleries.shape[0] == len(probe_feat)

        use_full_set = gallery_size == -1
        df = self.test_all.copy()

        # ====================formal=====================
        name_to_det_feat = {}
        for name, det, feat in zip(self.test_imnames,
                                   gallery_det, gallery_feat):
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
        # ret  # TODO: save json
        for i in range(len(probe_feat)):
            pid = i
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()
            # Ignore the probe image
            start = time.time()
            probe_imname = self.queries_to_galleries.iloc[i, 0]
            # probe_roi = df[df['imname'] == probe_imname]
            # probe_roi = probe_roi[probe_roi['is_query'] == 1]
            # probe_roi = probe_roi[probe_roi['pid'] == pid]
            # probe_roi = probe_roi.loc[:, 'x1': 'y2'].as_matrix()
            probe_gt = []
            tested = set([probe_imname])
            # 1. Go through the gallery samples defined by the protocol
            for g_i in range(1, gallery_size + 1):
                gallery_imname = self.queries_to_galleries.iloc[i, g_i]
                # gt = df[df['imname'] == gallery_imname]
                # gt = gt[gt['pid'] == pid]  # important
                # gt = gt.loc[:, 'x1': 'y2']
                gt = df.query('imname==@gallery_imname and pid==@pid')
                gt = gt.loc[:, 'x1': 'y2'].as_matrix().ravel()
                count_gt += (gt.size > 0)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat: continue
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
            y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])
            # compute time cost
            end = time.time()
            print('{}-th loop, cost {:.4f}s'.format(i, end - start))

        print('search ranking:')
        print('  mAP = {:.2%}'.format(np.mean(aps)))
        accs = np.mean(accs, axis=0)
        for i, k in enumerate(topk):
            print('  top-{:2d} = {:.2%}'.format(k, accs[i]))


if __name__ == '__main__':

    rtdir = '/Users/liliangqi/Desktop/person_search/dataset'
    size = 100
    sysu = PersonSearchDataset(rtdir, split_name='test', gallery_size=size)

    output_dir = './data/test_result'
    gboxes_file = open(os.path.join(output_dir, 'gboxes.pkl'), 'rb')
    gboxes = pickle.load(gboxes_file, encoding='iso-8859-1')
    gfeatures_file = open(os.path.join(output_dir, 'gfeatures.pkl'), 'rb')
    gfeatures = pickle.load(gfeatures_file, encoding='iso-8859-1')
    pfeatures_file = open(os.path.join(output_dir, 'pfeatures.pkl'), 'rb')
    pfeatures = pickle.load(pfeatures_file, encoding='iso-8859-1')
    # sysu.evaluate_detections(gboxes)
    sysu.evaluate_search(gboxes, gfeatures['feat'], pfeatures['feat'],
                         det_thresh=0.5, gallery_size=size, dump_json=None)
    gboxes_file.close()
    gfeatures_file.close()
    pfeatures_file.close()