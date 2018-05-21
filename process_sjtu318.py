# -----------------------------------------------------
# Generate Annotations for Person Search Dataset
#
# Author: Liangqi Li
# Creating Date: May 19, 2018
# Latest rectifying: May 20, 2018
# -----------------------------------------------------
import os
import os.path as osp
import shutil
import random
from collections import Counter

import numpy as np
import pandas as pd


def pick_dir(root_dir):
    """Pick out children directions from `root_dir`"""

    children_dirs = []
    for file_name in os.listdir(root_dir):
        abs_file_name = osp.join(root_dir, file_name)
        if osp.isdir(abs_file_name):
            children_dirs.append(abs_file_name)

    return children_dirs


def pick_mp4(root_dir):
    """Pick out mp4 files from `root_dir`"""

    videos = []
    for file_name in os.listdir(root_dir):
        abs_file_name = osp.join(root_dir, file_name)
        if abs_file_name[-4:] == '.mp4':
            videos.append(abs_file_name)

    return videos


def pick_query(root_dir):
    """Pick out jpg or files from `root_dir`"""

    queries = []
    for file_name in os.listdir(root_dir):
        abs_file_name = osp.join(root_dir, file_name)
        if abs_file_name[-4:] == '.jpg' or abs_file_name[-4:] == '.png':
            queries.append(abs_file_name)

    return queries


def pick_txt(root_dir):
    """Pick out txt files from `root_dir`"""

    annos = []
    for file_name in os.listdir(root_dir):
        abs_file_name = osp.join(root_dir, file_name)
        if abs_file_name[-4:] == '.txt':
            annos.append(abs_file_name)

    return annos


def rename_video_dir(root_dir):
    """Rename video dirs that contain jpg images"""

    datasets = pick_dir(root_dir)
    for dataset in datasets:
        persons = pick_dir(dataset)
        for person in persons:
            video_dirs = pick_dir(person)
            for i, video_dir in enumerate(video_dirs):
                os.rename(video_dir, os.path.join(person, str(i+1)))


def rename_query(root_dir):
    """Rename query images"""

    datasets = pick_dir(root_dir)
    for dataset in datasets:
        persons = pick_dir(dataset)
        for person in persons:
            queries = pick_query(person)
            for i, query in enumerate(queries):
                os.rename(query, os.path.join(
                    person, 'q_{}.{}'.format(i+1, query[-3:])))


def rename_img_and_txt(root_dir):
    """Rename images and annotation txt files"""

    datasets = pick_dir(root_dir)
    for dataset in datasets:
        persons = pick_dir(dataset)
        for person in persons:
            video_dirs = pick_dir(person)
            for video_dir in video_dirs:
                imgs = pick_query(video_dir)
                annos = pick_txt(video_dir)
                d_name = dataset.split('/')[-1][-1]
                p_name = person.split('/')[-1]
                v_name = video_dir.split('/')[-1]
                for img in imgs:
                    im_name = img.split('/')[-1][:-4]
                    os.rename(img, os.path.join(
                        video_dir, 'd{}_p{}_v{}_{}.jpg'.format(
                            d_name, p_name, v_name, im_name)))
                for anno in annos:
                    an_name = anno.split('/')[-1][:-4]
                    os.rename(anno, os.path.join(
                        video_dir, 'd{}_p{}_v{}_{}.txt'.format(
                            d_name, p_name, v_name, an_name)))


def collect_files(root_dir, dest_dir):
    """Collect images and annotation txt files"""

    img_dest_dir = os.path.join(dest_dir, 'SSM')
    anno_dest_dir = os.path.join(dest_dir, 'annotation')

    if not os.path.exists(img_dest_dir):
        os.mkdir(img_dest_dir)
    if not os.path.exists(anno_dest_dir):
        os.mkdir(anno_dest_dir)

    datasets = pick_dir(root_dir)
    for dataset in datasets:
        persons = pick_dir(dataset)
        for person in persons:
            video_dirs = pick_dir(person)
            for video_dir in video_dirs:
                imgs = pick_query(video_dir)
                annos = pick_txt(video_dir)
                for img in imgs:
                    im_name = img.split('/')[-1]
                    if os.path.join(video_dir, im_name[:-4] + '.txt') in annos:
                        dest_img = os.path.join(img_dest_dir, im_name)
                        shutil.copyfile(img, dest_img)
                for anno in annos:
                    an_name = anno.split('/')[-1]
                    dest_anno = os.path.join(anno_dest_dir, an_name)
                    shutil.copyfile(anno, dest_anno)


def collect_queries(root_dir, dest_dir):
    """Collect query images"""

    query_dest_dir = os.path.join(dest_dir, 'query')
    if not os.path.exists(query_dest_dir):
        os.mkdir(query_dest_dir)

    datasets = pick_dir(root_dir)
    for dataset in datasets:
        d_name = dataset.split('/')[-1][-1]
        persons = pick_dir(dataset)
        for person in persons:
            p_name = person.split('/')[-1]
            queries = pick_query(person)
            for query in queries:
                q_name = 'd{}_p{}_'.format(d_name, p_name) + \
                         query.split('/')[-1]
                dest_query = os.path.join(query_dest_dir, q_name)
                shutil.copyfile(query, dest_query)


def produce_train_and_test(root_dir):
    """Produce `trainAllDF.csv` and `trainImnamesSe.csv`"""

    img_dir = os.path.join(root_dir, 'SSM')
    anno_dir = os.path.join(root_dir, 'annotation')
    txt_dir = os.path.join(anno_dir, 'txt')

    train_box_imnames = []
    train_boxes = np.zeros((1, 5), dtype=np.int32)
    test_box_imnames = []
    test_boxes = np.zeros((1, 5), dtype=np.int32)

    dataset_to_pid = {1: 0, 2: 75, 3: 145, 4: 235, 5: -30}

    txt_annos = pick_txt(txt_dir)
    for anno in txt_annos:
        im_name = anno.split('/')[-1][:-4] + '.jpg'
        d_name = int(anno.split('/')[-1].split('_')[0][1:])
        p_name = int(anno.split('/')[-1].split('_')[1][1:])
        with open(anno, 'r') as f:
            num_box = int(f.readline().rstrip())
            for _ in range(num_box):
                line = f.readline().rstrip().split(' ')
                line = line[0:1] + line[2:]
                pid, x1, y1, x2, y2 = [int(i) for i in line]
                del_x = x2 - x1
                del_y = y2 - y1
                if pid == 1:
                    if d_name != 4:
                        pid = p_name - 1 + dataset_to_pid[d_name]
                    else:
                        if p_name in range(2, 10):
                            pid = p_name - 2 + dataset_to_pid[d_name]
                        else:
                            pid = p_name - 9 + dataset_to_pid[d_name]
                    if d_name != 5:
                        if pid > 199:
                            pid -= 3
                        elif pid > 182:
                            pid -= 2
                        elif pid > 174:
                            pid -= 1
                else:
                    pid = -1
                box = np.array([x1, y1, del_x, del_y, pid])
                if d_name != 5:
                    train_boxes = np.vstack((train_boxes, box))
                    train_box_imnames.append(im_name)
                else:
                    test_boxes = np.vstack((test_boxes, box))
                    test_box_imnames.append(im_name)

    # Remove the first row
    train_boxes = train_boxes[1:]
    test_boxes = test_boxes[1:]

    # Indicate the order of the column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'cls_id', 'pid']

    train_boxes_df = pd.DataFrame(
        train_boxes, columns=['x1', 'y1', 'del_x', 'del_y', 'pid'])
    train_boxes_df['imname'] = train_box_imnames
    train_boxes_df['cls_id'] = np.ones((train_boxes.shape[0], 1),
                                       dtype=np.int32)
    train_boxes_df = train_boxes_df[ordered_columns]

    test_boxes_df = pd.DataFrame(
        test_boxes, columns=['x1', 'y1', 'del_x', 'del_y', 'pid'])
    test_boxes_df['imname'] = test_box_imnames
    test_boxes_df['cls_id'] = np.ones((test_boxes.shape[0], 1), dtype=np.int32)
    test_boxes_df = test_boxes_df[ordered_columns]

    train_imnames = list(set(train_box_imnames))
    test_imnames = list(set(test_box_imnames))
    train_imnames = pd.Series(train_imnames)
    test_imnames = pd.Series(test_imnames)

    train_boxes_df.to_csv(os.path.join(anno_dir, 'trainAllDF.csv'),
                          index=False)
    test_boxes_df.to_csv(os.path.join(anno_dir, 'testAllDF.csv'), index=False)
    train_imnames.to_csv(os.path.join(anno_dir, 'trainImnamesSe.csv'),
                         index=False)
    test_imnames.to_csv(os.path.join(anno_dir, 'testImnamesSe.csv'),
                        index=False)


def produce_query_set(root_dir):
    """Produce query set"""

    anno_dir = os.path.join(root_dir, 'annotation')
    test_boxes_df = pd.read_csv(os.path.join(anno_dir, 'testAllDF.csv'))

    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'cls_id', 'pid',
                       'num_g']
    query_boxes_df = pd.DataFrame([['0', 0, 0, 0, 0, 0, 0, 0]],
                                  columns=ordered_columns)

    for pid in sorted(list(set(test_boxes_df['pid']))):
        if pid == -1:
            continue
        df = test_boxes_df[test_boxes_df['pid'] == pid]
        chosen = random.choice(range(df.shape[0]))
        query = df.iloc[chosen].copy()
        query['num_g'] = df.shape[0] - 1
        query = query.to_frame().transpose()
        query_boxes_df = pd.concat((query_boxes_df, query))

    query_boxes_df = query_boxes_df.iloc[1:, :]
    query_boxes_df.index = range(query_boxes_df.shape[0])
    query_boxes_df = query_boxes_df.drop(['cls_id'], axis=1)  # remove `cls_id`

    query_boxes_df.to_csv(os.path.join(anno_dir, 'queryDF.csv'), index=False)


def produce_query_gallery(root_dir):
    """Produce query_to_gallery"""

    anno_dir = os.path.join(root_dir, 'annotation')
    test_boxes_df = pd.read_csv(os.path.join(anno_dir, 'testAllDF.csv'))
    query_boxes_df = pd.read_csv(os.path.join(anno_dir, 'queryDF.csv'))
    test_imnames = pd.read_csv(os.path.join(anno_dir, 'testImnamesSe.csv'),
                               header=None, squeeze=True)

    max_num_g = max(query_boxes_df['num_g'])
    chosen_sizes = [50, 100, 200, 500]
    gallery_sizes = [size for size in chosen_sizes if size > max_num_g]

    for size in gallery_sizes:
        queries_to_galleries = [[] for _ in range(query_boxes_df.shape[0])]
        for i in range(query_boxes_df.shape[0]):
            q_name = query_boxes_df.iloc[i]['imname']
            pid = query_boxes_df.iloc[i]['pid']
            df = test_boxes_df[test_boxes_df['pid'] == pid]

            gt_gallery = list(set(df['imname']))
            gt_gallery.remove(q_name)
            for gt_im in gt_gallery:
                queries_to_galleries[i].append(gt_im)

            # Add other images that don't contain the `pid` person to fill
            candidates = list(set(test_imnames) - set(df['imname']))
            num_to_fill = size - len(queries_to_galleries[i])
            chosen_ones = random.sample(candidates, num_to_fill)
            queries_to_galleries[i].extend(chosen_ones)

        queries_to_galleries = pd.DataFrame(queries_to_galleries,
                                            index=query_boxes_df['imname'])
        queries_to_galleries.to_csv(os.path.join(
            anno_dir, 'q_to_g{}DF.csv'.format(size)))


def main():

    root_dir = '/Users/liliangqi/Desktop/myResearch/mydataset/label_result'
    dest_dir = '/Users/liliangqi/Desktop/myResearch/mydataset/sjtu318'
    # rename_video_dir(root_dir)
    # rename_query(root_dir)
    # rename_img_and_txt(root_dir)
    # collect_files(root_dir, dest_dir)
    # collect_queries(root_dir, dest_dir)
    produce_train_and_test(dest_dir)
    # produce_query_set(dest_dir)
    # produce_query_gallery(dest_dir)

    # test_all = pd.read_csv(osp.join(dest_dir, 'annotation', 'testAllDF.csv'))
    # test_imnames = pd.read_csv(osp.join(
    #     dest_dir, 'annotation', 'testImnamesSe.csv'),
    #     header=None, squeeze=True)
    # exception_test_image = []
    # for im_name in test_imnames:
    #     im_df = test_all[test_all['imname'] == im_name]
    #     counter = Counter(im_df['pid'])
    #     for id in counter.keys():
    #         if id > -1 and counter[id] > 1:
    #             exception_test_image.append(im_name)

    train_imnames = pd.read_csv(osp.join(
        dest_dir, 'annotation','trainImnamesSe.csv'),
        header=None, squeeze=True)
    train_all = pd.read_csv(osp.join(dest_dir, 'annotation', 'trainAllDF.csv'))
    exception_train_image = []
    for im_name in train_imnames:
        im_df = train_all[train_all['imname'] == im_name]
        counter = Counter(im_df['pid'])
        for id in counter.keys():
            if id > -1 and counter[id] > 1:
                exception_train_image.append(im_name)

    print('Debug')

if __name__ == '__main__':

    main()
