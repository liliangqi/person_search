# -----------------------------------------------------------
# Generate Annotations from SYSU for Person Search Dataset
#
# Author: Liangqi Li
# Creating Date: Mar 16, 2018
# Latest rectifying: Apr 28, 2018
# -----------------------------------------------------------

import os
import os.path as osp
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io as sio


def process_images_mat(root_dir):
    """
    process `Images.mat`
    ---
    return:
        bboxes_df: pd.DataFrame with shape (n_bbox, 6)
                   columns: [imname, x1, y1, del_x, del_y, cls_id]
        imnames: pd.Series with shape (n_images,)
    """
    imgs = sio.loadmat(osp.join(root_dir, 'Images.mat'))['Img'].squeeze()

    imnames = []
    imnames_unique = []
    all_bboxes = np.zeros((1, 4), dtype=np.int32)

    for im_name, _, boxes in imgs:
        imname = str(im_name[0])
        bboxes = np.asarray([b[0][0] for b in boxes[0]])
        valid_index = np.where((bboxes[:, 2] > 0) & (bboxes[:, 3] > 0))[0]
        assert valid_index.size > 0, \
            'Warning: {} has no valid boxes.'.format(imname)
        bboxes = bboxes[valid_index].astype(np.int32)
        all_bboxes = np.vstack((all_bboxes, bboxes))
        imnames.extend([imname] * bboxes.shape[0])
        imnames_unique.append(imname)

    all_bboxes = all_bboxes[1:]
    bboxes_df = pd.DataFrame(all_bboxes,
                             columns=['x1', 'y1', 'del_x', 'del_y'])
    bboxes_df['imname'] = imnames
    bboxes_df = bboxes_df[['imname', 'x1', 'y1', 'del_x', 'del_y']]
    bboxes_df['cls_id'] = np.ones((all_bboxes.shape[0], 1), dtype=np.int32)
    # drop duplicated values of all bboxes
    bboxes_df = bboxes_df.drop_duplicates()
    bboxes_df.index = range(bboxes_df.shape[0])

    imnames = pd.Series(imnames_unique)

    bboxes_df.to_csv(osp.join(root_dir, 'bboxesDF.csv'), index=False)
    imnames.to_csv(osp.join(root_dir, 'imnamesSe.csv'), index=False)

    return bboxes_df, imnames


def process_train_mat(root_dir):
    """
    process `test/train_test/Train.mat`
    ---
    return:
        train_df: pd.DataFrame with shape (n_bbox, 7)
                  columns: [imname, x1, y1, del_x, del_y, cls_id, pid]
                  all boxes in it are labeled samples of training set
    """
    train = sio.loadmat(osp.join(root_dir, 'test/train_test/Train.mat'))
    train = train['Train'].squeeze()

    train_imnames = []
    train_bboxes = np.zeros((1, 4), dtype=np.int32)
    pids = []

    for index, item in enumerate(train):
        scenes = item[0, 0][2].squeeze()
        for im_name, box, _ in scenes:
            imname = str(im_name[0])
            bbox = box.squeeze().astype(np.int32)
            train_bboxes = np.vstack((train_bboxes, bbox))
            train_imnames.append(imname)
            pids.append(index)

    train_bboxes = train_bboxes[1:]
    pids = np.array(pids, dtype=np.int32)
    train_df = pd.DataFrame(train_bboxes,
                            columns=['x1', 'y1', 'del_x', 'del_y'])
    train_df['imname'] = train_imnames
    train_df = train_df[['imname', 'x1', 'y1', 'del_x', 'del_y']]
    train_df['cls_id'] = np.ones((train_df.shape[0], 1), dtype=np.int32)
    train_df['pid'] = pids
    # drop duplicated values of all bboxes
    train_df = train_df.drop_duplicates()
    train_df.index = range(train_df.shape[0])

    train_df.to_csv(osp.join(root_dir, 'trainLabeledDF.csv'), index=False)

    return train_df


def process_pool_mat(root_dir):
    """
    process `pool.mat`
    ---
    return:
        train: pd.Series with shape (n_train_ims,)
               contains all imnames for training
        test: pd.Series with shape (n_test_ims,)
              contains all imnames for testing
    """
    test = sio.loadmat(osp.join(root_dir, 'pool.mat'))
    test = test['pool'].squeeze()
    test = pd.Series([str(a[0]) for a in test])

    all_imgs = pd.read_csv(osp.join(root_dir, 'imnamesSe.csv'), header=None,
                           squeeze=True)
    train = list(set(all_imgs) - set(test))
    train = pd.Series(train)

    train.to_csv(osp.join(root_dir, 'trainImnamesSe.csv'), index=False)
    test.to_csv(osp.join(root_dir, 'testImnamesSe.csv'), index=False)

    return train, test


def produce_train_all(root_dir):
    """
    Produce a DataFrame to save all boxes for training and save it as a csv
    file. Then set pid labels to these boxes. Unlabeled samples will be set -1
    as their pid labels.
    ---
    return:
        train_all: pd.DataFrame with shape (n_train_box, 7)
                   columns: [imname, x1, y1, del_x, del_y, cls_id, pid]
                   each row is a bbox that contains a person
    """
    train = 'trainImnamesSe.csv'
    all_bboxes = 'bboxesDF.csv'
    train_lab = 'trainLabeledDF.csv'

    if train in os.listdir(root_dir) and all_bboxes in os.listdir(root_dir) \
            and train_lab in os.listdir(root_dir):
        train_imnames = pd.read_csv(osp.join(root_dir, train),
                                    header=None, squeeze=True)
        all_bboxes = pd.read_csv(osp.join(root_dir, all_bboxes))
        train_lab = pd.read_csv(osp.join(root_dir, train_lab))
    else:
        train_imnames, _ = process_pool_mat(root_dir)
        all_bboxes, _ = process_images_mat(root_dir)
        train_lab = process_train_mat(root_dir)

    # pick out items whose `imname` appear in `train_imnames`
    # to form training set. NOTE there is another way to finish it
    # ```
    # train_all = all_bboxes[all_bboxes['imname'].isin(train_imnames)]
    # train_all.index = range(train_all.shape[0])
    # ```
    train_all = pd.merge(all_bboxes, train_imnames.to_frame(name='imname'))
    train_all = pd.merge(train_all, train_lab, how='outer')  # set pids
    train_all = train_all.fillna(-1)  # set -1 to unlabeled samples
    train_all['pid'] = train_all['pid'].values.astype(np.int32)

    train_all.to_csv(osp.join(root_dir, 'trainAllDF.csv'), index=False)

    return train_all


def process_test_mat(root_dir):
    """
    process `test/train_test/TestG50.mat`
    ---
    return:
        test_df: pd.DataFrame with shape (n_test_bbox, 8)
                 columns: [imname, x1, y1, del_x, del_y, cls_id, pid, is_query]
                 all boxes in it are labeled samples of training set
        queries_to_galleries: pd.DataFrame with shape (n_query, 50)
                              index: image names of queries
    """
    test = sio.loadmat(osp.join(root_dir, 'test/train_test/TestG50.mat'))
    test = test['TestG50'].squeeze()

    test_imnames = []
    q_names = []
    test_bboxes = np.zeros((1, 4), dtype=np.int32)
    pids = []
    is_query = []
    queries_to_galleries = [[] for _ in range(len(test))]

    for index, item in enumerate(test):
        # query
        q_name = str(item['Query'][0, 0][0][0])
        q_box = item['Query'][0, 0][1].squeeze().astype(np.int32)
        test_imnames.append(q_name)
        test_bboxes = np.vstack((test_bboxes, q_box))
        pids.append(index)
        is_query.append(1)
        q_names.append(q_name)
        # gallery
        gallery = item['Gallery'].squeeze()
        for im_name, box, _ in gallery:
            g_name = str(im_name[0])
            queries_to_galleries[index].append(g_name)
            if box.size == 0:
                continue
            g_box = box.squeeze().astype(np.int32)
            test_imnames.append(g_name)
            test_bboxes = np.vstack((test_bboxes, g_box))
            pids.append(index)
            is_query.append(0)

    test_bboxes = test_bboxes[1:]
    pids = np.array(pids, dtype=np.int32)
    is_query = np.array(is_query, dtype=np.int32)
    test_df = pd.DataFrame(test_bboxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    test_df['imname'] = test_imnames
    test_df = test_df[['imname', 'x1', 'y1', 'del_x', 'del_y']]
    test_df['cls_id'] = np.ones((test_df.shape[0], 1), dtype=np.int32)
    test_df['pid'] = pids
    test_df['is_query'] = is_query

    queries_to_galleries = pd.DataFrame(queries_to_galleries, index=q_names)

    test_df.to_csv(osp.join(root_dir, 'testLabeledDF.csv'), index=False)
    queries_to_galleries.to_csv(osp.join(root_dir, 'q_to_g50DF.csv'))

    return test_df, queries_to_galleries


def produce_test_all(root_dir):
    """
    Produce a DataFrame to save all boxes for testing and save it as a csv
    file. Then set pid labels and `is_query` flag to these boxes. Unlabeled
    samples will be set -1 as their pid labels.
    ---
    return:
        test_all: pd.DataFrame with shape (n_test_box, 8)
                  column: [imname, x1, y1, del_x, del_y, cls_id, pid, is_query]
                  each row is a bbox that contains a person
    """
    test = 'testImnamesSe.csv'
    all_bboxes = 'bboxesDF.csv'
    test_lab = 'testLabeledDF.csv'

    if test in os.listdir(root_dir) and all_bboxes in os.listdir(root_dir) \
            and test_lab in os.listdir(root_dir):
        test_imnames = pd.read_csv(osp.join(root_dir, test),
                                   header=None, squeeze=True)
        all_bboxes = pd.read_csv(osp.join(root_dir, all_bboxes))
        test_lab = pd.read_csv(osp.join(root_dir, test_lab))
    else:
        _, test_imnames = process_pool_mat(root_dir)
        all_bboxes, _ = process_images_mat(root_dir)
        test_lab, _ = process_test_mat(root_dir)

    test_all = all_bboxes[all_bboxes['imname'].isin(test_imnames)]
    test_all.index = range(test_all.shape[0])
    test_all = pd.merge(test_all, test_lab, how='outer')
    test_all['pid'] = test_all['pid'].fillna(-1)
    test_all['is_query'] = test_all['is_query'].fillna(0)
    test_all['pid'] = test_all['pid'].values.astype(np.int32)
    test_all['is_query'] = test_all['is_query'].values.astype(np.int32)

    test_all.to_csv(osp.join(root_dir, 'testAllDF.csv'), index=False)

    return test_all


def produce_gallery_set(root_dir,
                        gallery_sizes=[50, 100, 500, 1000, 2000, 4000]):
    """
    produce several DataFrames to save reflects from queries to galleries,
    image names of queries are used as index
    """
    for size in gallery_sizes:
        file_name = 'TestG' + str(size)
        test = sio.loadmat(osp.join(root_dir, 'test/train_test',
                                    file_name + '.mat'))
        test = test[file_name].squeeze()

        q_names = []
        queries_to_galleries = [[] for _ in range(len(test))]
        for index, item in enumerate(test):
            # query
            q_name = str(item['Query'][0, 0][0][0])
            q_names.append(q_name)
            # gallery
            gallery = item['Gallery'].squeeze()
            for im_name, _, __ in gallery:
                g_name = str(im_name[0])
                queries_to_galleries[index].append(g_name)

        queries_to_galleries = pd.DataFrame(queries_to_galleries,
                                            index=q_names)
        queries_to_galleries.to_csv(osp.join(root_dir,
                                             'q_to_g' + str(size) + 'DF.csv'))


def produce_query_set(root_dir):
    """produce query set"""

    test = sio.loadmat(osp.join(root_dir, 'test/train_test', 'TestG50.mat'))
    test = test['TestG50'].squeeze()
    test_all = pd.read_csv(osp.join(root_dir, 'testAllDF.csv'))

    q_names = []
    q_boxes = np.zeros((1, 4), dtype=np.int32)
    pids = []

    for index, item in enumerate(test):
        q_name = str(item['Query'][0, 0][0][0])
        q_names.append(q_name)
        q_box = item['Query'][0, 0][1].squeeze().astype(np.int32)
        q_boxes = np.vstack((q_boxes, q_box))
        pids.append(index)

    # Indicate hte order of column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'pid']
    q_boxes = q_boxes[1:]
    q_boxes_df = pd.DataFrame(q_boxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    q_boxes_df['imname'] = q_names
    q_boxes_df['pid'] = pids
    q_boxes_df = q_boxes_df[ordered_columns]

    q_boxes_df.to_csv(osp.join(root_dir, 'queryDF.csv'), index=False)


def main():

    rtdir = '/Users/liliangqi/Desktop/person_search/dataset/annotation'
    # bbox_df, imnames_se = process_images_mat(rtdir)

    # bbox_df = pd.read_csv(osp.join(rtdir, 'bboxesDF.csv'))
    # imnames_se = pd.read_csv(osp.join(rtdir, 'imnamesSe.csv'), squeeze=True)

    # train_labeled_df = process_train_mat(rtdir)

    # process_pool_mat(rtdir)

    # produce_train_all(rtdir)

    # process_test_mat(rtdir)

    # produce_test_all(rtdir)

    # produce_gallery_set(rtdir)

    produce_query_set(rtdir)

    test_all = pd.read_csv(osp.join(rtdir, 'testAllDF.csv'))
    test_imnames = pd.read_csv(osp.join(rtdir, 'testImnamesSe.csv'),
                               header=None, squeeze=True)
    exception_test_image = []
    for im_name in test_imnames:
        im_df = test_all[test_all['imname'] == im_name]
        counter = Counter(im_df['pid'])
        for id in counter.keys():
            if id > -1 and counter[id] > 1:
                exception_test_image.append(im_name)

    train_imnames = pd.read_csv(osp.join(rtdir, 'trainImnamesSe.csv'),
                                header=None, squeeze=True)
    train_all = pd.read_csv(osp.join(rtdir, 'trainAllDF.csv'))
    exception_train_image = []
    for im_name in train_imnames:
        im_df = train_all[train_all['imname'] == im_name]
        counter = Counter(im_df['pid'])
        for id in counter.keys():
            if id > -1 and counter[id] > 1:
                exception_train_image.append(im_name)


if __name__ == '__main__':

    main()