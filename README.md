 # Network Deconvolution

The source paper: https://arxiv.org/abs/1905.11926

The source code: https://github.com/yechengxi/deconvolution

Our team verified the conclusion of the paper using part of the code provided by the author and our own simple implement, our conclusion is below: 

The experimental results are consistent with the authors' conclusion no matter in simple CNN or complex CNN. Network Deconvolution can at least achieve the effect of Batch Norm, and usually can improve the accuracy of the image classification tasks. Meanwhile, it can use a relatively large learning rate like BN, the ND has a higher convergence speed than BN as well. As the number of network layers increases and the structure becomes more complex, due to considering the pixel correlation(or covariance computation), the ND is much more time-consuming, although the authors implemented the so-called Fast Deconvolution. We noticed that there are still many normalization methods proposed after BN, the authors only compare ND and BN which is relatively narrow. In addition, after reading the relevant articles, our analysis concluded that the original authors were closer to batch whitening rather than normalization. If we can expand on this in the future, we hope to conduct a more in-depth comparative analysis with other variants.

 ## Dependencies

- python == 3.6


- pytorch == 1.6.0


- tensorflow == 2.3.0


- CUDA == 10.1

 ## Settings Overview

 According to the author's instructions, the model can be run by the command. 

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

For example:

```
python main.py --arch vgg11 --dataset cifar100 --batch-size 128 --epochs 50 --lr 0.1 --wd 0.01 --deconv True --block-fc 512 --loss CE --optimizer SGD
```

