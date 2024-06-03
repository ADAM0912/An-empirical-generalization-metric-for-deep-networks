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
You might use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs, and/or switch to CIFAR10 by `--dataset cifar10`.  
**(1) linear probe**

First, Apply linear probe to your model and save the result for future use. 
```
python linear_probe.py --batch_size 64 \
  --learning_rate 1e-4 \
```

**(2) Calculate accuracy and kappa**  
You need to modify the file location and different setting according to your situation. 
```
python  calculate_accuracy.py
python  calculate_kappa.py
```
**(3) Find the Tradeoff point and output the bound** 
You need to modify the file location and different setting according to your situation. 
```
python  trade_off_point.py

```
The final result of our benchmark is like this(change clip to your own testing model):

| MODEL             | CLIP     |
|-------------------|----------|
| GENERALIZATION BOUND | 0.308677 |
| DIVERSITY BOUND   | 0.293403 |
| SSIM              | 0.65     |
| ZERO-SHOT%        | 0        |
| MODEL SIZE        | 38M      |

