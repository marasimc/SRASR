# Syntax-Aware Retrieval Augmentation for Neural Symbolic Regression

## Overview

SRASR is a syntax-aware retrieval-augmented framework for symbolic regression. It retrieves structure-relevant tokens and adaptively fuses them with neural predictions, yielding superior performance over standard models.

## Environment Setup

```bash
conda create -n srasr python=3.10
conda activate srasr
# conda install faiss-cpu -c pytorch # For CPU
conda install faiss-gpu -c pytorch # For CUDA

pip install -r requirements.txt
```

## Run the Model

```shell
cd knnbox-scripts/enhance_knn_sr

# step 1. build datastore
bash build_datastore.sh

# step 2. train metak-network and confidence network
bash train_network.sh

# step 3. run inference
# note: before running inference, please make sure you have already placed the benchmark dataset in `symbolicregression_utils/datasets/` (relative to the working directory)
bash run.sh
```

## Acknowledgements

This work is mainly refers to the following codebases. We sincerely thank the authors for open-sourcing their excellent work.

* [kNN-Box](https://github.com/NJUNLP/knn-box)
* [symbolicregression](https://github.com/facebookresearch/symbolicregression)
