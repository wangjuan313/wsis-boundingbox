# Bounding Box Tightness Prior for Weakly Supervised Image Segmentation

This project hosts the codes for the implementation of the paper **Bounding Box Tightness Prior for Weakly Supervised Image Segmentation** (MICCAI 2021) [[miccai](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_49)] [[arxiv](https://arxiv.org/abs/2110.00934)].



# Dataset preprocessing

Download [Promise12](https://promise12.grand-challenge.org/) dataset, and put it on the "data/prostate" folder.

Download [Atlas](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) dataset, and put it on the "data/atlas" folder.

Run the following codes for preprocessing:

```bash
# trainig and valid subsets for promise12 dataset
python preprocess/slice_promise_train.py
python preprocess/slice_promise_valid.py

# trainig and valid subsets for atlas dataset
python preprocess/slice_atlas.py
```

# Training

```bash
#  The following experiments include full supervision (exp_no=0), MIL ablation study (exp_no=1), smooth maximum approximation ablation study (exp_no=2,3), and main experiments (exp_no=4,5)

# training for promise12 dataset, exp_no=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python tools/train_promise_unetwithbox.py --n_exp exp_no
# training for atlas dataset, exp_no=0,1,2,3,4,5
CUDA_VISIBLE_DEVICES=0 python tools/train_atlas_unetwithbox.py --n_exp exp_no
```

# Validation

```bash
# Dice validation results for promise12 dataset, exp_no=0,1,2,...,16
CUDA_VISIBLE_DEVICES=0 python tools/valid_promise_unetwithbox.py --n_exp exp_no
# Dice validation results for atlas dataset, exp_no=0,1,2,...,16
CUDA_VISIBLE_DEVICES=0 python tools/valid_atlas_unetwithbox.py --n_exp exp_no
```

# Performance summary

```bash
python tools/report_promise_unetwithbox_paper.py
python tools/report_atlas_unetwithbox_paper.py
```

## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{wang2021bounding,
  title={Bounding Box Tightness Prior for Weakly Supervised Image Segmentation},
  author={Wang, Juan and Xia, Bin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={526--536},
  year={2021},
  organization={Springer}
}
```

## Logs
1. 3/16/2022: a bug in _C_promise.py was fixed such that the training subset was used for training.


