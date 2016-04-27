# CSCI-544-Final-Project
 Final Project for CSCI-544 Spring 2016

# Instructions

To run the whole experiment run: (approx. 20min on a recent NVIDIA GPU)

```
make all
```

It should download the dataset, unzip, train and serialize the core components, and finally train and benchmark the augmented, multimodal model.

A typical output should be: 

```
> python experiment/data.py
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-09.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-10.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-07-25.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-07-24.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-08.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-01.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-07-22.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-04.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-11.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-06.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-07-27.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-07-26.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-07-21.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-07-23.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-02.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-07.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-08-03.zip
> ./SaudiNewsNet/SaudiNewsNet-master/dataset/2015-07-31.zip
> vocabulary size -  343172
> # of samples -  31030
> # of classes 14
> class distribution -  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13] [2080 2947 2964 3690 4852 2279 2090 3065  133 2846 1411 2252   52  369]
> sentence length -  1527 [   0    1    2 ..., 4480 5160 6200] [354  42 104 ...,   1   1   1]
> # of train - 24740, # of valid - 6290
> vocabulary from dataset is saved into labeledTrainData.tsv.vocab
> python experiment/train_core.py -e 2
> 2016-04-26 23:11:08,573 - neon.backends - WARNING - deterministic_update and deterministic args are deprecated in favor of specifying random seed
> Epoch 0   [Train |████████████████████|  194/194  batches, 2.17 cost, 65.13s]
> Epoch 1   [Train |████████████████████|  193/193  batches, 1.16 cost, 63.86s]
> Train accuracy:  [ 0.90553761]
> Test accuracy:  [ 0.49904612]
> python experiment/train_augmented.py -e 7 -i 1
> 2016-04-26 23:14:20,195 - neon.backends - WARNING - deterministic_update and deterministic args are deprecated in favor of specifying random seed
> Epoch 0   [Train |████████████████████|  194/194  batches, 0.71 cost, 124.93s]
> Epoch 1   [Train |████████████████████|  193/193  batches, 0.45 cost, 124.21s]
> Epoch 2   [Train |████████████████████|  193/193  batches, 0.34 cost, 124.83s]
> Epoch 3   [Train |████████████████████|  194/194  batches, 0.30 cost, 127.64s]
> Epoch 4   [Train |████████████████████|  193/193  batches, 0.25 cost, 126.71s]
> Epoch 5   [Train |████████████████████|  193/193  batches, 0.23 cost, 126.60s]
> Epoch 6   [Train |████████████████████|  193/193  batches, 0.24 cost, 126.76s]
> Train accuracy:  [ 0.96604687]
> Test accuracy:  [ 0.49252781]
```

# License
The content of this repository is exclusive to the purpose of CSCI 544 class at
USC. Copy or redistribution of the code is strictly forbidden without prior
authorization of all the authors.
