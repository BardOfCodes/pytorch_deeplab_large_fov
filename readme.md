# Pytorch Deeplab Large-FOV

This repository contains training, testing and conversion scripts for using Deeplab Large-FOV (introduced in[this paper](https://arxiv.org/abs/1412.7062)), in pytorch.

## Set-up

An excellent resource for setting up PASCAL VOC 2012 task, is [this repository](https://github.com/martinkersner/train-DeepLab). By following the instructions there, you can:<br>

* Find links for downloading the PASCAL VOC 2012 dataset, and PASCAL VOC Augmented dataset.
* Convert Ground truth files to an easier to access format.

To use the same initialization as the authors used, download the initialization caffemodel files [here](http://liangchiehchen.com/projects/Init%20Models.html) (download 'vgg16_20M.caffemodel').

### Converting caffemodel to usable OrderedDictionary

There are two ways of doing this,

1) If you have caffe installed: You can easily use the converter.py script.

2) If you do not have caffe(or dont want to install it): You can try to use code from [this repository](https://github.com/ethereon/caffe-tensorflow)  

For converting using converter.py, use:

```
python converter.py <caffe_path> <model_caffemodel> <model_prototxt>
```

where, `<caffe_path>`, `<model_caffemodel>` and `<model_prototxt>` are paths to the respestive entities.


## Training

There are two training regime for Deeplab Large-FOV version 1,

* 1: use 'step' learning policy.
  * batch size: 20
  * maximum iterations: 6000
  * learning rate decay policy: multiply by 0.1 every 2000 iterations. 

* 2: use 'poly' learning policy.
  * batch size: 10
  * maximum iterations: 20000
  * learning rate decay policy: multiply by $\big( (1- \frac{iter}{max_iter})^{power} \big)$ every iteration.

for training the networks, use:

```
python train_v1.py <list_path> <im_path> <gt_path>
or
python train_v2.py <list_path> <im_path> <gt_path>
```

where, 
* `<list_path>` is the path to the text_file which consist of names training image(For eg: train_aug.txt) ,
* `<im_path>` is the path to the folder containing image files, and 
* `<gt_path>`is the path to the folder containing ground-truth files.

For additional options, use:
```
python train_v1.py -h
```

## Testing

For saving the output of your trained network, use
```
python test.py <model_path> <im_path> <im_list> <save_path>
```
where,

* `<model_path>` is the path to your saved model,
* `<im_path>` is the path to the folder containing test images,
* `<im_list>` is the path to file containing, and
* `<save_path>` is the path to folder where you want to save the output.


For additional options, use:
```
python test.py -h
```

## Evaluation

For evaluation you can use code [here](https://github.com/BardOfCodes/seg_metrics_pytorch). It has been linked as a submodule of this repository.

Use:
```
python demo.py find_metrics predict_path gt_path id_file
```
where,
* `predict_path` is the path to folder containing predicted segmentation maps.
* `gt_path` is the path to folder containing ground truth segmentation maps.
* `id_file` is the path to file with image names.

## Results

The performance of Deeplab Large-FOV trained in pytorch is different from it's performance when trained in caffe (which can be validated by following [@martinkersner/train-DeepLab](https://github.com/martinkersner/train-DeepLab)).

The following table list the mean IOU accross the 21 classes in Pascal VOC 2012. The results are for the 'val' set.

| **Training Regime** | **Training Regime 1** | **Training Regime 2** |
|---|---|---|
|**With Dropout layers** | 61.576 % | 65.22 % |
|**Without Dropout layers** | 63.570 % | 65.65 % |
|**From Caffe** | **2.25 %** | **65.88 %**|

** Important Note: Dropout Layers are being disabled by using `model.eval()`, instead of `model.train()` in the 'train_v1/2.py' scripts.**

## Acknowlegdement

I would like to thank :

* [@martinkersner](https://github.com/martinkersner) for making setup in caffe very easy,
* [@gaurav_pandey](https://discuss.pytorch.org/u/gaurav_pandey/summary) for showing the correct implementation of deeplab's learning policy, 
* [@isht7](https://github.com/isht7) for making [this repository](https://github.com/isht7/pytorch-deeplab-resnet). My code is based on this.

Also, a big thanks to [Video Analytics Lab](http://val.serc.iisc.ernet.in/valweb/).