# A Testbed for Assessing Generalization Capacity of Deep Neural Networks

This repository presents the code implementation of the paper '[A practical generalization metric for deep networks benchmarking](https://arxiv.org/pdf/2409.01498)'.

Benchmarking is conducted on the CIFAR-100 dataset, with only 50 randomly selected object classes used for training. The remaining classes are reserved for zero-shot testing.

Each pre-trained model is fine-tuned on CIFAR-100 to adapt itself to this task. Subsequently, various model variants are created by adding a single linear probe layer for subsequent zero-shot performance evaluation.

These models are then tested, collecting metrics such as error rate and kappa. Finally, the trade-off point between these metrics is calculated and visualized in graphs, ready for analysis.
<p align="center">
  <img src="figures/benchbed.png" width="700">
</p>


## Prerequisites 

| | Command | Notes |
| - | - | - |
| Pytorch | `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` |Depending on the GPU configuration, go to https://pytorch.org for more details|
| scikit-learn| `conda create -n sklearn-env -c conda-forge scikit-learn` `conda activate sklearn-env`| |
| EfficientNet_PyTorch | `pip install efficientnet_pytorch` | |
| Pandas | `pip install pandas` | |
| utils | `pip install utils` |  |
| h5py  | `pip install h5py` | |
| SciPy | `pip install scipy`| |
| Matplotlib| `pip install matplotlib`| |


## Running

For better navigability, the process is divided into multiple scripts.

Use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs, and/or switch to imagenet by `--dataset 
imagenet`.  

**Step 1. Model Preparetion**

To enable zero-shot testing, a linear probe layer is added to the tuned model to match the desired output dimension. The resulting linear layers are then saved.

```
python linear_probe.py
```

**Step 2. Evaluating( ErrorRate and Kappa )**

The performance of pre-trained CLIP and EfficientNet models is evaluated across three dimensions: zero-shot percentage, weight number, and Structural Similarity Index (SSIM). 

The error rates and Kappas for each class are recorded in the corresponding cells of a 3D array.

```
python calculate_error.py
python calculate_kappa.py
```

The following script calculates three types of statistics for the distributions of Error Rate and Kappa across all classes: mean, standard deviation, and 10th percentile. These statistics are updated cell-wise within the 3D array. Please note that you may need to adjust file locations and other settings in the script based on your setup. The final results are saved in an XLSX format file. 

```
python  build_3d_array.py
```

**Step 3.Benchmark Result Calculation( Tradeoff point and The bound )** 

The trade-off points are calculated using Equation 4 from the paper. It will return you three value which show the tradeoff point in SSIM, ZEROSHOT, Model Size. 
```
python  tradeoff_point.py
```

Enter these points in the plot_marginal_distribution.py and then visualized based on the three pairs of marginal distributions, as described by Equation 5.
```
python  plot_marginal_distribution.py
```

<p align="center">
  <img src="figures/example_graph.png" width="1001">
</p>

## Benchmarking Result

The final result of our benchmark(CLIP model in cifar100) are presented below.
|  **Dataset**            |                    | **ImageNet** |       | **CIFAR-100** |       |
|--------------|--------------------|:------------:|:-----:|:-------------:|:-----:|
| **Model Type** |                    | **CLIP**     | **EFFICIENT NET** | **CLIP**  | **EFFICIENT NET** |
| **Generalization Bound** | | 0.279         | 0.284 | 0.657          | 0.600 |
| **Diversity Bound**      | | 0.276         | 0.280 | 0.668          | 0.608 |
| **SSIM (lower bound)**   | | 0.874         | 0.805 | 0.949          | 0.937 |
| **ZEROSHOT (upper bound)** | | 0.285     | 0.295 | 0.182          | 0.258 |
| **Model Size (lower bound)** | | 116M      | 20M  | 163M           | 22M  |





