# ----------------------------------------------------------
# Generate Annotations from PRW for Person Search Dataset
#
# Author: Liangqi Li
# Creating Date: Apr 26, 2018
# Latest rectifying: Apr 28, 2018
# ----------------------------------------------------------
import os
import random

import numpy as np
import pandas as pd
import scipy.io as sio


def process_annotations(root_dir):
    """Process all annotation MAT files"""

    annotation_dir = os.path.join(root_dir, 'annotations')
    file_names = sorted(os.listdir(annotation_dir))

    train_imnames = sio.loadmat(os.path.join(root_dir, 'frame_train.mat'))
    train_imnames = train_imnames['img_index_train'].squeeze()
    test_imnames = sio.loadmat(os.path.join(root_dir, 'frame_test.mat'))
    test_imnames = test_imnames['img_index_test'].squeeze()
    train_imnames = [train_name[0] + '.jpg' for train_name in train_imnames]
    test_imnames = [test_name[0] + '.jpg' for test_name in test_imnames]

    train_box_imnames = []
    train_boxes = np.zeros((1, 5), dtype=np.int32)
    test_box_imnames = []
    test_boxes = np.zeros((1, 5), dtype=np.int32)

    for f_name in file_names:
        im_name = f_name[:-4]
        f_dir = os.path.join(annotation_dir, f_name)
        boxes = sio.loadmat(f_dir)
        if 'box_new' in boxes.keys():
            boxes = boxes['box_new']
        elif 'anno_file' in boxes.keys():
            boxes = boxes['anno_file']
        elif 'anno_previous' in boxes.keys():
            boxes = boxes['anno_previous']
        else:
            raise KeyError(boxes.keys()[-1])
        valid_index = np.where((boxes[:, 3] > 0) & (boxes[:, 4] > 0))[0]
        assert valid_index.size > 0, \
            'Warning: {} has no valid boxes.'.format(im_name)
        boxes = boxes[valid_index].astype(np.int32)

        if im_name[:-4] in train_imnames:
            train_boxes = np.vstack((train_boxes, boxes))
            train_box_imnames.extend([im_name] * boxes.shape[0])
        elif im_name[:-4] in test_imnames:
            test_boxes = np.vstack((test_boxes, boxes))
            test_box_imnames.extend([im_name] * boxes.shape[0])
        else:
            print('{:s} does not exsit'.format(im_name))
            exit()

    # remove the first row
    train_boxes = train_boxes[1:]
    test_boxes = test_boxes[1:]

    # Indicate the order of the column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'cls_id', 'pid']

    train_boxes_df = pd.DataFrame(
        train_boxes, columns=['pid', 'x1', 'y1', 'del_x', 'del_y'])
    train_boxes_df['imname'] = train_box_imnames
    train_boxes_df['cls_id'] = np.ones((train_boxes.shape[0], 1),
                                       dtype=np.int32)
    train_boxes_df = train_boxes_df[ordered_columns]

    test_boxes_df = pd.DataFrame(
        test_boxes, columns=['pid', 'x1', 'y1', 'del_x', 'del_y'])
    test_boxes_df['imname'] = test_box_imnames
    test_boxes_df['cls_id'] = np.ones((test_boxes.shape[0], 1), dtype=np.int32)
    test_boxes_df = test_boxes_df[ordered_columns]

    train_imnames = pd.Series(train_imnames)
    test_imnames = pd.Series(test_imnames)

    train_boxes_df.to_csv(os.path.join(root_dir, 'trainAllDF.csv'),
                          index=False)
    test_boxes_df.to_csv(os.path.join(root_dir, 'testAllDF.csv'), index=False)
    train_imnames.to_csv(os.path.join(root_dir, 'trainImnamesSe.csv'),
                         index=False)
    test_imnames.to_csv(os.path.join(root_dir, 'testImnamesSe.csv'),
                        index=False)


def fix_train_test(root_dir):
    """Fix training set and test set."""

    train_boxes_df = pd.read_csv(os.path.join(root_dir, 'trainAllDF.csv'))
    test_boxes_df = pd.read_csv(os.path.join(root_dir, 'testAllDF.csv'))
    train_imnames = pd.read_csv(os.path.join(root_dir, 'trainImnamesSe.csv'),
                                header=None, squeeze=True)
    test_imnames = pd.read_csv(os.path.join(root_dir, 'testImnamesSe.csv'),
                               header=None, squeeze=True)
    train_ids = sio.loadmat(os.path.join(
        root_dir, 'ID_train.mat'))['ID_train'].squeeze()
    test_ids = sio.loadmat(os.path.join(
        root_dir, 'ID_test.mat'))['ID_test2'].squeeze()

    # Pick out those images that do NOT contain test IDs
    remove_imnames = []
    only_unlabeled_imnames = []  # TODO: may be we can add these to test set

    for im_name in test_imnames:
        df = test_boxes_df[test_boxes_df['imname'] == im_name]
        if not set(df['pid']) & set(test_ids):
            remove_imnames.append(im_name)
            if set(df['pid']) == {-2}:
                only_unlabeled_imnames.append(im_name)

    train_imnames = set(train_imnames) | set(remove_imnames)
    train_imnames = pd.Series(list(train_imnames))
    test_imnames = set(test_imnames) - set(remove_imnames)
    test_imnames = pd.Series(list(test_imnames))

    for im_name in remove_imnames:
        df = test_boxes_df[test_boxes_df['imname'] == im_name]
        train_boxes_df = pd.concat([train_boxes_df, df])
        test_boxes_df.drop(df.index, inplace=True)

    train_boxes_df.index = range(train_boxes_df.shape[0])
    test_boxes_df.index = range(test_boxes_df.shape[0])

    train_dict = {id_num: i for i, id_num in enumerate(train_ids)}
    test_dict = {id_num: i for i, id_num in enumerate(test_ids)}

    # Change those IDs not in test_ids to -1
    for i in range(test_boxes_df.shape[0]):
        if test_boxes_df.ix[i, 'pid'] not in test_ids:
            test_boxes_df.ix[i, 'pid'] = -1
        else:
            test_boxes_df.ix[i, 'pid'] = test_dict[test_boxes_df.ix[i, 'pid']]

    # Change pid with value -2 in training set to -1
    for i in range(train_boxes_df.shape[0]):
        if train_boxes_df.ix[i, 'pid'] == -2:
            train_boxes_df.ix[i, 'pid'] = -1
        else:
            train_boxes_df.ix[i, 'pid'] = train_dict[
                train_boxes_df.ix[i, 'pid']]

    # Rewirte csv files
    train_boxes_df.to_csv(os.path.join(root_dir, 'trainAllDF.csv'),
                          index=False)
    test_boxes_df.to_csv(os.path.join(root_dir, 'testAllDF.csv'), index=False)
    train_imnames.to_csv(os.path.join(root_dir, 'trainImnamesSe.csv'),
                         index=False)
    test_imnames.to_csv(os.path.join(root_dir, 'testImnamesSe.csv'),
                        index=False)


def produce_query_set(root_dir):
    """Produce query set"""

    test_ids = sio.loadmat(os.path.join(
        root_dir, 'ID_test.mat'))['ID_test2'].squeeze()
    test_dict = {id_num: i for i, id_num in enumerate(test_ids)}

    query_imnames = []  # TODO: change is_query in dataset.py
    query_boxes = np.zeros((1, 5), dtype=np.int32)

    with open(os.path.join(root_dir, 'query_info.txt')) as f:
        for line in f.readlines():
            line = line.split(' ')
            im_name = line[-1].rstrip() + '.jpg'
            box = np.array([line[:-1]]).astype(np.float32).astype(np.int32)
            query_imnames.append(im_name)
            query_boxes = np.vstack((query_boxes, box))

    query_boxes = query_boxes[1:]

    # Indicate the order of the column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'pid']
    query_boxes_df = pd.DataFrame(
        query_boxes, columns=['pid', 'x1', 'y1', 'del_x', 'del_y'])
    query_boxes_df['imname'] = query_imnames
    query_boxes_df = query_boxes_df[ordered_columns]

    for i in range(query_boxes_df.shape[0]):
        query_boxes_df.ix[i, 'pid'] = test_dict[query_boxes_df.ix[i, 'pid']]

    query_boxes_df.to_csv(os.path.join(root_dir, 'queryDF.csv'), index=False)


def produce_query_gallery(root_dir):
    """Produce query_to_gallery"""

    test_ids = sio.loadmat(os.path.join(
        root_dir, 'ID_test.mat'))['ID_test2'].squeeze()
    test_boxes_df = pd.read_csv(os.path.join(root_dir, 'testAllDF.csv'))
    query_boxes_df = pd.read_csv(os.path.join(root_dir, 'queryDF.csv'))
    test_imnames = pd.read_csv(os.path.join(root_dir, 'testImnamesSe.csv'),
                               header=None, squeeze=True)

    # Count how many images that contain the specific ID
    id_appearence = {}
    for id_num in test_ids:
        num_boxes = test_boxes_df[test_boxes_df['pid'] == id_num].shape[0]
        id_appearence[id_num] = num_boxes

    # get gallery sizes
    chosen_sizes = [50, 100, 200, 500, 1000, 2000, 4000]
    gallery_sizes = [size for size in chosen_sizes
                     if size > max(id_appearence.values())]

    for size in gallery_sizes:
        queries_to_galleries = [[] for _ in range(query_boxes_df.shape[0])]
        for i in range(query_boxes_df.shape[0]):
            q_name = query_boxes_df.iloc[i]['imname']
            q_camera = q_name[1]
            pid = query_boxes_df.iloc[i]['pid']
            df = test_boxes_df[test_boxes_df['pid'] == pid]

            # gt_gallery refers to those images that contain the `pid` person
            gt_gallery = list(set(df['imname']))
            gt_gallery.remove(q_name)
            for gt_im in gt_gallery:
                # Only pick out images under different cameras with query
                if gt_im[1] != q_camera:
                    queries_to_galleries[i].append(gt_im)

            # Add other images that don't contain the `pid` person to fill
            candidates = list(set(test_imnames) - set(df['imname']))
            num_to_fill = size - len(queries_to_galleries[i])
            chosen_ones = random.sample(candidates, num_to_fill)
            queries_to_galleries[i].extend(chosen_ones)

        queries_to_galleries = pd.DataFrame(queries_to_galleries,
                                            index=query_boxes_df['imname'])
        queries_to_galleries.to_csv(os.path.join(
            root_dir, 'q_to_g' + str(size) + 'DF.csv'))


def main():
    root_dir = '/Users/habor/Desktop/myResearch/PRW4person_search/prw_orig'
    # process_annotations(root_dir)
    # fix_train_test(root_dir)
    # produce_query_set(root_dir)
    produce_query_gallery(root_dir)

if __name__ == '__main__':

    main()
