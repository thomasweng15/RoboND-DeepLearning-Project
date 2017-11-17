Follow Me Project Writeup
===

## Data 

I trained using the default test and validation data provided for us, and then gathered some data in simulation of the hero moving in dense crowds to supplement the existing data.

## Network Architecture

diagrma

### Layers

My network used three encoder layers, one 1x1 convolution, and three decoder layers, as shown in the diagram above. 

#### Encoder blocks

The encoder layers 
did what
had what strides and filter sizes and why
had what convolutional batchnorms and why

#### 1x1 Convolution



#### Decoder blocks

The decoder layers 
did what
had what skip layer concatenations and why

### Hyperparameters

The learning rate needs to be small enough to prevent the model from overshooting optimal model parameters when applying gradient descent during backpropagation, but it also needs to be large enough so that the model actually reaches the optimum in a reasonable amount of time. I played with the learning rate and observed the loss values for the training and validation sets until I found a well-paced learning rate of 0.01 with largely monotonic loss value reductions. 

The batch size determines how many data samples to train before running a backpropagation step. A small batch size 

High epoch

Mid batch size, Mid steps per epoch

## Notes from Training 

I used an AWS EC2 instance to train my models. 

I trained my model using the provided training and validation sets, reaching a score of 0.399.

Then I retrained my model using 42MB of data I collected in simulation observing the hero moving in dense crowds. This data was used as another test set. I used the same validation set from the first run. 

## Limitations

This trained model is only able to follow one "hero" human. It is unable to generalize to hero humans that don't look like the one in the training data.

The model also cannot generalize to do image segmentation on other entities besides humans, e.g. cars, animals, and other objects. 

A new model would have be trained with data labeled for a new hero or a new type of entity in order to address these limitations. 
