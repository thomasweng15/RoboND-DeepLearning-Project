Follow Me Project Writeup
===

## Data 

default, flip, add some

## Network Architecture

diagrma

### Layers

3 encoder layers, 1 1x1 convolution, 3 decoder layers

#### Encoder blocks

Use two conv 2d batchnorms per block

#### 1x1 Convolution

#### Decoder blocks

Skip layer concatenation

### Hyperparameters

Played with learning rate until good-paced learning with no overshoot

High epoch

Mid batch size, Mid steps per epoch

## Notes from Training 

Used AWS instance 

Training set

Validation set

Collected data in sim but didn't use it

## Limitations

Doesn't work on other things other than humans

Need labeled data of a hero person or other object
