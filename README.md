# boundary_detection

Based on https://github.com/Lavender105/DFF

## Setup

## Preprocess Cityscapes

```Bash
python preprocess/preproc_original.py
```

## Training

```Bash
CUDA_VISIBLE_DEVICES=0,1 python boundary/train.py --checkname dff_test --base-size 640 --crop-size 640 --worker 4 --batch-size 2
```

## Test

```Bash
```

## Evaluate

Currently, MATLAB is necessary to run the evaluation code.
