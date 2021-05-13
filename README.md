 # Network Deconvolution

The source paper: https://arxiv.org/abs/1905.11926

The source code: https://github.com/yechengxi/deconvolution

 ## Dependencies

Python: 3.6

Pytorch: 1.6.0

Tensorflow: 2.3.0

CUDA: 10.1

 ## Settings Overview
 We have included a few settings you can add into the run command.

 The basic run command (for non-imagenet dataset) is:

 ```
 python main.py --[keyword1] [argument1] --[keyword2] [argument2]  ...
 ```

 The major keywords to note are:

 * deconv - set to True or False if you want to test deconv (True) or BN (False)
 * arch - use a given architecture (resnet50, vgg11, vgg13, vgg19, densenet121)
 * wd - sets the weight decay to a given value
 * batch-size - sets the batch size
 * epochs - the number of epochs to run
 * dataset - the dataset to use (cifar10, cifar100) 
 * lr - sets the learning rate
 * block - block size in deconvolution
 * block-fc - block size in decorrelating the fully connected layers.