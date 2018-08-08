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

3. Generate appropriate annotation files for the datasets. The path to the specific dataset is denoted as DATASET_PATH. *This step may take a long while, please be patient.*

   ```bash
   cd dataset
   ```

   - CUHK-SYSU

     ```bash
     mv DATASET_PATH/Image/SSM DATASET_PATH/frames
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

4. (Optional) Get pre-trained ResNet-50 [here](https://drive.google.com/file/d/14ZsYYXq6t9mv_2BMMuEFLcCeZaXLMWAF/view?usp=sharing).

## Train

Train the model by indicating the model name, the path to the dataset and dataset name ("sysu" or "prw")

```bash
python train.py --net res50 --data_dir DATASET_PATH --dataset_name sysu
```

## Test

The test code is rewriting now.

