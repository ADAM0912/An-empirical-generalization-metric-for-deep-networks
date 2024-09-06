#Benchmarking Test for deep network generalization

This repo covers an implementation for the Paper An empirical generalization metric to benchmark deep networks, using CIFAR as an example. 

## Introduction of the benchmark
Currently, most efforts to estimate the generalization error bounds are enduring, there is a growing interest in intuitive metrics to measure generalization capacity experimentally. This is not only a practical request, and also important for theoretical research. This is because theoretical estimations must be verified in practice. Particularly, there is a lack of research on benchmarking various deep network generalisation capacity and verifying theoretical estimations.  This paper aims to introduce an empirical generalization metric for benchmarking various deep networks, and proposes a novel testbed for verifying theoretical estimations. Our observation underscores that a deep network's generalization capacity in classical classification scenarios hinges on both classification accuracy and the diversity of unseen data. The proposed metric system can quantify model accuracy and data diversity, offering an intuitive and quantitative assessment, that is, trade-off point. Moreover, we compare our empirical metric with the existing generalization estimations through our benchmarking testbed. Unfortunately, most of the available generalization estimations don't align with the practical measurements using our proposed empirical metric. The illustration of Benchmarking bed is shown below:
<p align="center">
  <img src="figures/benchbed.png" width="700">
</p>


## Installation
To set up the environment:
```
conda env create -f environment.yml
```

## Running
You might use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs, and/or switch to imagenet by `--dataset imagenet`.  
**Step 1. Collect ErrorRate and Kappa data of your test models**

In our example, we test the pretrained models of CLIP and EfficientNet on test data across three dimensions (i.e., zero-shot%, weight
number, SSIM) and store the error rates and Kappas for each class in each cell of a 3D array.
```
python train.py \
python linear_probe.py \
python calculate_error.py \
python calculate_kappa.py \
```

**(2) Step 2. Update 3D Array**  

We compute three kinds of statistics related to the distributions of ErrorRate and Kappa across all classes, i.e., means,standard derivations, 10th percentiles, and update them cell-
wise in the 3D array.You need to modify the file location and different setting according to your situation and the final result is saved as a xlsx file. 
```
python  build_3d_array.py
```
**(3) Step 3.Find the Tradeoff point and output the bound** 

We compute the trade-off points by Eq.4 in the paper and visualize the trade-off points by Eq.5 in the paper based on three pairs of marginal distributions, 
```
python  tradeoff_point.py
python  plot_marginal_distribution.py

```
The final result of our benchmark is like this(CLIP model in cifar100):

| MODEL             | CLIP     |
|-------------------|----------|
| GENERALIZATION BOUND | 0.852 |
| DIVERSITY BOUND   | 0.164    |
| SSIM              | 0.824    |
| ZERO-SHOT%        | 0.228    |
| MODEL SIZE        | 56M      |

