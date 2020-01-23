# Source code

## Contents ##
* loss.py : custom loss functions
  * weakly_supervised_sparse_categorical_crossentropy : proposed loss function
* model.py : Modified U-Net

## Annotation format ###
Weakly annotated pixels are supposed to have offsetted pixel values.
i.e. $l_{\overline{c}} = l_{c} + N_{classes}$

In the case 3 classes segmentation
* $l_{c1} = 0$ -> $l_{\overline{c1}} = 3$
* $l_{c2} = 1$ -> $l_{\overline{c2}} = 4$
* $l_{c3} = 2$ -> $l_{\overline{c3}} = 5$

## Test environment ##
* Ubuntu 18.04
* Python 3.5.2
* Keras 2.2.4
* tensorflow 1.14.0
