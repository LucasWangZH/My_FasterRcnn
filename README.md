The offical Faster R-CNN code written in python is [here](https://github.com/rbgirshick/py-faster-rcnn)

## 1.Intro
To enhance my understanding of anchor based object detection by realzing Faster-Rcnn.

The official code has too many encapsulation and is hard to read.

I try to comb through the main process and realize it in my way. Thanks to the strong help of [chenyuntc](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)

The code includes data process, train and evaluation phases.

Annotation includes some chinese when I am coding.

## 2.Performace
dataset:@0.5, VOC 2007 test
GPU: one P-4000

#### map
map basically achieves 0.67. 2 points lower than official.

#### speed
inference faster than 198ms per picture average in 100 images.

## 3.Requirements
Need pytorch v1.0




Thanks to this [repo](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) from chenyuntc, which helps me a lot in this reproduction.
