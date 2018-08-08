# -----------------------------------------------------------
# Generate Annotations from SYSU for Person Search Dataset
#
# Author: Liangqi Li
# Creating Date: Mar 16, 2018
# Latest rectifying: Aug 8, 2018
# -----------------------------------------------------------

import os
from collections import Counter

import numpy as np
import pandas as pd
import argparse
import scipy.io as sio


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Prepare the dataset.')
    parser.add_argument('--dataset_dir', default='', type=str)
    args = parser.parse_args()

    return args


def process_annotations(root_dir, save_dir):
    """
    Process `Images.mat`
    ---
    return:
        bboxes_df: pd.DataFrame with shape (n_bbox, 6)
                   columns: [imname, x1, y1, del_x, del_y, cls_id]
        imnames: pd.Series with shape (n_images,)
    """
    annotation_dir = os.path.join(root_dir, 'annotation')
    imgs = sio.loadmat(os.path.join(
        annotation_dir, 'Images.mat'))['Img'].squeeze()

    imnames = []
    imnames_unique = []
    all_bboxes = []

    for im_name, _, boxes in imgs:
        imname = str(im_name[0])
        bboxes = np.asarray([b[0][0] for b in boxes[0]])
        valid_index = np.where((bboxes[:, 2] > 0) & (bboxes[:, 3] > 0))[0]
        assert valid_index.size > 0, \
            'Warning: {} has no valid boxes.'.format(imname)
        bboxes = bboxes[valid_index].astype(np.int32)
        all_bboxes.append(bboxes)
        imnames.extend([imname] * bboxes.shape[0])
        imnames_unique.append(imname)

    # Concat all the boxes
    all_bboxes = np.vstack(all_bboxes).astype(np.int32)
    # Indicate the order of the column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'cls_id']
    bboxes_df = pd.DataFrame(
        all_bboxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    bboxes_df['imname'] = imnames
    bboxes_df['cls_id'] = np.ones((all_bboxes.shape[0], 1), dtype=np.int32)
    bboxes_df = bboxes_df[ordered_columns]
    # Drop duplicated values of all bboxes
    bboxes_df = bboxes_df.drop_duplicates()
    bboxes_df.index = range(bboxes_df.shape[0])

    imnames = pd.Series(imnames_unique)
    train_imnames, test_imnames = process_pool_mat(annotation_dir, imnames)
    train_lb_df = process_train_mat(annotation_dir)
    test_lb_df = process_test_mat(annotation_dir)
    train_boxes_df = produce_split_all(train_imnames, bboxes_df, train_lb_df)
    test_boxes_df = produce_split_all(test_imnames, bboxes_df, test_lb_df)
    train_test_dfs = (train_boxes_df, test_boxes_df, train_imnames,
                      test_imnames)
    train_test_dfs = remove_outliers(train_test_dfs)

    # Write csv files
    train_boxes_df, test_boxes_df, train_imnames, test_imnames = train_test_dfs
    train_imnames.to_csv(os.path.join(
        save_dir, 'trainImnamesSe.csv'), index=False)
    test_imnames.to_csv(os.path.join(
        save_dir, 'testImnamesSe.csv'), index=False)
    train_boxes_df.to_csv(os.path.join(
        save_dir, 'trainAllDF.csv'), index=False)
    test_boxes_df.to_csv(os.path.join(save_dir, 'testAllDF.csv'), index=False)


def process_train_mat(annotation_dir):
    """
    process `test/train_test/Train.mat`
    ---
    return:
        train_df: pd.DataFrame with shape (n_bbox, 7)
                  columns: [imname, x1, y1, del_x, del_y, cls_id, pid]
                  all boxes in it are labeled samples of training set
    """
    train = sio.loadmat(os.path.join(
        annotation_dir, 'test/train_test/Train.mat'))
    train = train['Train'].squeeze()

    train_imnames = []
    train_bboxes = []
    pids = []

    for index, item in enumerate(train):
        scenes = item[0, 0][2].squeeze()
        for im_name, box, _ in scenes:
            imname = str(im_name[0])
            bbox = box.squeeze().astype(np.int32)
            train_bboxes.append(bbox)
            train_imnames.append(imname)
            pids.append(index)

    # Concat all the boxes
    train_bboxes = np.vstack(train_bboxes).astype(np.int32)
    # Indicate the order of the column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'cls_id', 'pid']
    pids = np.array(pids, dtype=np.int32)
    train_lb_df = pd.DataFrame(
        train_bboxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    train_lb_df['imname'] = train_imnames
    train_lb_df['cls_id'] = np.ones((train_lb_df.shape[0], 1), dtype=np.int32)
    train_lb_df['pid'] = pids
    train_lb_df = train_lb_df[ordered_columns]
    # Drop duplicated values of all bboxes
    train_lb_df = train_lb_df.drop_duplicates()
    train_lb_df.index = range(train_lb_df.shape[0])

    return train_lb_df


def process_pool_mat(annotation_dir, all_imnames):
    """
    process `pool.mat`
    ---
    return:
        train: pd.Series with shape (n_train_ims,)
               contains all imnames for training
        test: pd.Series with shape (n_test_ims,)
              contains all imnames for testing
    """
    test = sio.loadmat(os.path.join(annotation_dir, 'pool.mat'))
    test = test['pool'].squeeze()
    test = pd.Series([str(a[0]) for a in test])
    train = list(set(all_imnames) - set(test))
    train = pd.Series(train)

    return train, test


def produce_split_all(imnames, all_bboxes, lb_df):
    """
    Produce a DataFrame to save all boxes for training and testing, and save it
    as a csv file. Then set pid labels to these boxes. Unlabeled samples will
    be set -1 as their pid labels.
    ---
    return:
        train_all: pd.DataFrame with shape (n_train_box, 7)
                   columns: [imname, x1, y1, del_x, del_y, cls_id, pid]
                   each row is a bbox that contains a person
    """
    split_all = pd.merge(all_bboxes, imnames.to_frame(name='imname'))
    split_all = pd.merge(split_all, lb_df, how='outer')  # set pids
    split_all = split_all.fillna(-1)  # set -1 to unlabeled samples
    split_all['pid'] = split_all['pid'].values.astype(np.int32)

    return split_all


def process_test_mat(annotation_dir):
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
    test = sio.loadmat(os.path.join(
        annotation_dir, 'test/train_test/TestG50.mat'))
    test = test['TestG50'].squeeze()

    test_imnames = []
    test_bboxes = []
    pids = []

    for index, item in enumerate(test):
        # Query
        q_name = str(item['Query'][0, 0][0][0])
        q_box = item['Query'][0, 0][1].squeeze().astype(np.int32)
        test_imnames.append(q_name)
        test_bboxes.append(q_box)
        pids.append(index)
        gallery = item['Gallery'].squeeze()
        for im_name, box, _ in gallery:
            g_name = str(im_name[0])
            if box.size == 0:
                continue
            g_box = box.squeeze().astype(np.int32)
            test_imnames.append(g_name)
            test_bboxes.append(g_box)
            pids.append(index)

    # Concat all the boxes
    test_bboxes = np.vstack(test_bboxes).astype(np.int32)
    # Indicate the order of the column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'cls_id', 'pid']
    pids = np.array(pids, dtype=np.int32)
    test_lb_df = pd.DataFrame(
        test_bboxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    test_lb_df['imname'] = test_imnames
    test_lb_df['cls_id'] = np.ones((test_lb_df.shape[0], 1), dtype=np.int32)
    test_lb_df['pid'] = pids
    test_lb_df = test_lb_df[ordered_columns]

    # Drop duplicated values of all bboxes
    test_lb_df = test_lb_df.drop_duplicates()
    test_lb_df.index = range(test_lb_df.shape[0])

    return test_lb_df


def remove_outliers(train_test_dfs):
    """Remove the outlier images that contain more than one same person."""

    print('\nRemoving outliers in annotation files')
    train_boxes_df, test_boxes_df, train_imnames, test_imnames = train_test_dfs

    # Test set
    exception_test_indices = []
    for im_name in test_imnames:
        im_df = test_boxes_df[test_boxes_df['imname'] == im_name]
        counter = Counter(im_df['pid'])
        for idx in counter.keys():
            if idx > -1 and counter[idx] > 1:
                pid_df = im_df.query('pid==@idx')
                exc_inds = pid_df.index.tolist()[1:]
                exception_test_indices.extend(exc_inds)
    test_boxes_df.drop(exception_test_indices, axis=0, inplace=True)
    test_boxes_df.index = range(test_boxes_df.shape[0])

    # Training set
    exception_train_indices = []
    for im_name in train_imnames:
        im_df = train_boxes_df[train_boxes_df['imname'] == im_name]
        counter = Counter(im_df['pid'])
        for idx in counter.keys():
            if idx > -1 and counter[idx] > 1:
                pid_df = im_df.query('pid==@idx')
                exc_inds = pid_df.index.tolist()[1:]
                exception_train_indices.extend(exc_inds)
    train_boxes_df.drop(exception_train_indices, axis=0, inplace=True)
    train_boxes_df.index = range(train_boxes_df.shape[0])

    train_test_dfs = (train_boxes_df, test_boxes_df, train_imnames,
                      test_imnames)
    print('Done.\n')

    return train_test_dfs


def produce_query_set(root_dir, save_dir):
    """Produce query set"""

    annotation_dir = os.path.join(root_dir, 'annotation')
    test = sio.loadmat(os.path.join(
        annotation_dir, 'test/train_test', 'TestG50.mat'))
    test = test['TestG50'].squeeze()

    q_names = []
    q_boxes = []
    pids = []
    queries_num_g = []

    for index, item in enumerate(test):
        q_name = str(item['Query'][0, 0][0][0])
        gallery = item['Gallery'].squeeze()
        q_names.append(q_name)
        q_box = item['Query'][0, 0][1].squeeze().astype(np.int32)
        q_boxes.append(q_box)
        pids.append(index)
        num_g = 0
        for g_name, bbox, _ in gallery:
            if bbox.size > 0:
                num_g += 1
            else:
                break
        queries_num_g.append(num_g)

    # Indicate the order of column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'pid', 'num_g']
    q_boxes = np.vstack(q_boxes)
    q_boxes_df = pd.DataFrame(q_boxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    q_boxes_df['imname'] = q_names
    q_boxes_df['pid'] = pids
    q_boxes_df['num_g'] = queries_num_g
    q_boxes_df = q_boxes_df[ordered_columns]

    q_boxes_df.to_csv(os.path.join(save_dir, 'queryDF.csv'), index=False)


def produce_query_gallery(root_dir, save_dir):
    """
    Produce several DataFrames to save reflects from queries to galleries,
    image names of queries are used as indices
    """

    annotation_dir = os.path.join(root_dir, 'annotation')
    gallery_sizes = [50, 100, 500, 1000, 2000, 4000]

    for size in gallery_sizes:
        print('Producing gallery with size {}...'.format(size))
        file_name = 'TestG{}'.format(size)
        test = sio.loadmat(os.path.join(
            annotation_dir, 'test/train_test', '{}.mat'.format(file_name)))
        test = test[file_name].squeeze()

        q_names = []
        queries_to_galleries = [[] for _ in range(len(test))]
        for index, item in enumerate(test):
            # Query
            q_name = str(item['Query'][0, 0][0][0])
            q_names.append(q_name)
            # Gallery
            gallery = item['Gallery'].squeeze()
            for im_name, _, __ in gallery:
                g_name = str(im_name[0])
                queries_to_galleries[index].append(g_name)

        queries_to_galleries = pd.DataFrame(queries_to_galleries,
                                            index=q_names)
        queries_to_galleries.to_csv(os.path.join(
            save_dir, 'q_to_g{}DF.csv'.format(size)))


def main():

    args = parse_args()
    root_dir = args.dataset_dir
    save_dir = os.path.join(root_dir, 'SIPN_annotation')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Processing the mat files...')
    process_annotations(root_dir, save_dir)
    print('Producing test files')
    produce_query_set(root_dir, save_dir)
    produce_query_gallery(root_dir, save_dir)

    print('Dataset processing done.')


if __name__ == '__main__':

    main()
