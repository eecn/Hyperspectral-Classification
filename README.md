# DeepHyperX

A Python tool to perform deep learning experiments on various hyperspectral datasets.


## Description
The original code forked from GitLib project [Link](https://gitlab.inria.fr/naudeber/DeepHyperX).  
And there is a repository on GitHub, which maybe is the official project code. [DeepHyperX](https://github.com/nshaud/DeepHyperX)  
This repository will also update in the feature for research exchange.

## Requirements

This tool is compatible with Python 2.7 and Python 3.5+.

It is based on the [PyTorch](http://pytorch.org/) deep learning and GPU computing framework and use the [Visdom](https://github.com/facebookresearch/visdom) visualization server.

## Setup

The easiest way to install this code is to create a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) and to install dependencies using:
`pip install -r requirements.txt`

## Hyperspectral datasets

Several public hyperspectral datasets are available on the [UPV/EHU](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) wiki. Users can download those beforehand or let the tool download them. The default dataset folder is `./Datasets/`, although this can be modified at runtime using the `--folder` arg.

At this time, the tool automatically downloads the following public datasets:
  * Pavia University
  * Pavia Center
  * Kennedy Space Center
  * Indian Pines
  * Botswana

The [Data Fusion Contest 2018 hyperspectral dataset]() is also preconfigured, although users need to download it on the [DASE](http://dase.ticinumaerospace.com/) website and store it in the dataset folder under `DFC2018_HSI`.

An example dataset folder has the following structure:
```
Datasets
├── Botswana
│   ├── Botswana_gt.mat
│   └── Botswana.mat
├── DFC2018_HSI
│   ├── 2018_IEEE_GRSS_DFC_GT_TR.tif
│   ├── 2018_IEEE_GRSS_DFC_HSI_TR
│   ├── 2018_IEEE_GRSS_DFC_HSI_TR.aux.xml
│   ├── 2018_IEEE_GRSS_DFC_HSI_TR.HDR
├── IndianPines
│   ├── Indian_pines_corrected.mat
│   ├── Indian_pines_gt.mat
├── KSC
│   ├── KSC_gt.mat
│   └── KSC.mat
├── PaviaC
│   ├── Pavia_gt.mat
│   └── Pavia.mat
└── PaviaU
    ├── PaviaU_gt.mat
    └── PaviaU.mat
```

### Adding a new dataset

Adding a custom dataset can be done by modifying the `custom_datasets.py` file. Developers should add a new entry to the `CUSTOM_DATASETS_CONFIG` variable and define a specific data loader for their use case.

## Models

Currently, this tool implements several SVM variants from the [scikit-learn](http://scikit-learn.org/stable/) library and many state-of-the-art deep networks implemented in PyTorch.
  * SVM (linear, RBF and poly kernels with grid search)
  * SGD (linear SVM using stochastic gradient descent for fast optimization)
  * baseline neural network (4 fully connected layers with dropout)
  * 1D CNN ([Deep Convolutional Neural Networks for Hyperspectral Image Classification, Hu et al., Journal of Sensors 2015](https://www.hindawi.com/journals/js/2015/258619/))
  * Semi-supervised 1D CNN ([Autoencodeurs pour la visualisation d'images hyperspectrales, Boulch et al., GRETSI 2017](https://delta-onera.github.io/publication/2017-GRETSI))
  * 2D CNN ([Hyperspectral CNN for Image Classification & Band Selection, with Application to Face Recognition, Sharma et al, technical report 2018](https://lirias.kuleuven.be/bitstream/123456789/566754/1/4166_final.pdf))
  * Semi-supervised 2D CNN ([A semi-supervised Convolutional Neural Network for Hyperspectral Image Classification, Liu et al, Remote Sensing Letters 2017](https://www.tandfonline.com/doi/abs/10.1080/2150704X.2017.1331053))
  * 3D CNN ([3-D Deep Learning Approach for Remote Sensing Image Classification, Hamida et al., TGRS 2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565))
  * 3D FCN ([Contextual Deep CNN Based Hyperspectral Classification, Lee and Kwon, IGARSS 2016](https://arxiv.org/abs/1604.03519))
  * 3D CNN ([Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks, Chen et al., TGRS 2016](http://elib.dlr.de/106352/2/CNN.pdf))
  * 3D CNN ([Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network, Li et al., Remote Sensing 2017](http://www.mdpi.com/2072-4292/9/1/67))
  * 3D CNN ([HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image, Luo et al, ICPR 2018](https://arxiv.org/abs/1802.10478))
  * Multi-scale 3D CNN ([Multi-scale 3D Deep Convolutional Neural Network for Hyperspectral Image Classification, He et al, ICIP 2017](https://ieeexplore.ieee.org/document/8297014/))

### Adding a new model

Adding a custom deep network can be done by modifying the `models.py` file. This implies creating a new class for the custom deep network and altering the `get_model` function.

## Usage

Start a Visdom server:
`python -m visdom.server`
and go to [`http://localhost:8097`](http://localhost:8097) to see the visualizations (or [`http://localhost:9999`](http://localhost:9999) if you use Docker).

Then, run the script `main.py`.

The most useful arguments are:
  * `--model` to specify the model (e.g. 'svm', 'nn', 'hamida', 'lee', 'chen', 'li'),
  * `--dataset` to specify which dataset to use (e.g. 'PaviaC', 'PaviaU', 'IndianPines', 'KSC', 'Botswana'),
  * the `--cuda` switch to run the neural nets on GPU. The tool fallbacks on CPU if this switch is not specified.

There are more parameters that can be used to control more finely the behaviour of the tool. See `python main.py -h` for more information.

Examples:
  * `python main.py --model SVM --dataset IndianPines --training_sample 0.3`
    This runs a grid search on SVM on the Indian Pines dataset, using 30% of the samples for training and the rest for testing. Results are displayed in the visdom panel.
  * `python main.py --model nn --dataset PaviaU --training_sample 0.1 --cuda 0`
    This runs on GPU a basic 4-layers fully connected neural network on the Pavia University dataset, using 10% of the samples for training.
  * `python main.py --model hamida --dataset PaviaU --training_sample 0.5 --patch_size 7 --epoch 50 --cuda 0` 
    This runs on GPU the 3D CNN from Hamida et al. on the Pavia University dataset with a patch size of 7, using 50% of the samples for training and optimizing for 50 epochs.

