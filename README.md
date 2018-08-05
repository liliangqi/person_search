# Spatial Invariant Person Search Network

## Paper

Liangqi Li, Hua Yang, Lin Chen. "Spatial Invariant Person Search Network", PRCV 2018.

## Dataset

- CUHK-SYSU

  Please contact sli@ee.cuhk.edu.hk to require for this dataset.

- Person Re-identification in the Wild (PRW)

  Get the dataset [here](http://www.liangzheng.com.cn/Project/project_prw.html).

- SJTU318

  This is our own dataset and coming soon.

## Requirements

- Python 3.6
- CUDA 8.0 & cuDNN v5.1 **or** CUDA 9.1 & cuDNN v7.0
- PyTorch 0.4.0 or later
- torchvision 0.2.1or later
- OpenCV (python only is ok)

## Preparation

1. Clone this repository to the path called SIPN_ROOT.

   ```bash
   git clone https://github.com/liliangqi/person_search.git
   cd SIPN_ROOT
   ```

2. Compile the NMS module.

   ```bash
   sh setup.sh
   ```

3. Generate appropriate annotation files for the datasets. The path to the specific dataset is denoted as DATASET_PATH.

   ```bash
   cd dataset
   ```

   - CUHK-SYSU

     ```bash
     python process_sysu.py --dataset_dir DATASET_PATH
     ```

   - PRW

     ```bash
     python process_prw.py --dataset_dir DATASET_PATH
     ```

   - SJTU318 (coming soon)

     ```bash
     python process_sjtu318.py --dataset_dir DATASET_PATH
     ```

   ```
   cd ..
   ```

4. ​

Outliers in SYSU training set (one id appears twice in an image):
  ['s14319.jpg', 's1663.jpg', 's9239.jpg', 's4531.jpg', 's10716.jpg']

Outliers in SYSU testing set (one id appears twice in an image):
  ['s14430.jpg', 's3981.jpg', 's11888.jpg', 's888.jpg', 's5957.jpg', 's12295.jpg']

Outliers in PRW training set (one id appears twice in an image):
  ['c1s1_019476.jpg', 'c3s1_096067.jpg', 'c1s2_024091.jpg', 'c2s1_095446.jpg', 'c2s1_091696.jpg', 'c1s1_047526.jpg', 'c1s1_005401.jpg', 'c3s1_069842.jpg', 'c1s2_019916.jpg', 'c6s1_000476.jpg']

There are no outliers in PRW testing set.
