Follow Me Project Writeup
===

## Summary

I used an AWS EC2 instance to train my models. 

I trained my model using the provided training and validation sets, reaching a score of 0.399. Here's the markdown file for my model_training jupyter notebook output up to this point: [model_training.md]("./model_training/model_training.md")

I then attempted to use recorded data from the simulator to retrain my model to improve the score. However, I had small samples and had issues with overfitting to the new test data. 

Then I created flipped versions of all images and masks in the test and validation sets, doubling the amount of samples I had. I used that data to retrain my existing model on 15 epochs with the same hyperparameters, resulting in a score of 0.48. Here is the final HTML file with code and output: [model_training_FINAL.md]("./model_training_FINAL/model_training.html")s

Here are some screenshots from running the follower.py script with my model:
![screenshot1]("./screenshot1.png")
![screenshot2]("./screenshot2.png")
![screenshot3]("./screenshot3.png")

## Data 

I trained using the default test and validation data provided for us, and then gathered some data in simulation of the hero moving in dense crowds to supplement the existing data.

## Network Architecture

![diagram]("./model.png")

We want to preserve spatial information in the follow me task because we don't want to just identify the hero in the scene but also where he or she is in the image so we can follow. While fully connected networks might have convolutional layers that feed into a non-convolutional softmax layer, for example, a fully convolution network uses convolutions in every layer so that spatial information is maintained. As a result, I use a fully convolutional network consisting of three encoder layers, one 1x1 convolution, and three decoder layers, as shown in the diagram above. See the following sections for more detail about each section. 

#### Encoder blocks

The encoder blocks are comprised of convolutional layers that are trained to recognize features relevant in the input image for our follow me task. 

Each encoder block is structured similarly, with a separable convolutional layer that uses a 3x3 kernel to focuses on different regions of an input shape to learn features in each region. The kernel sweeps over the input with a stride of 2 in each block. This stride reduces the number of parameters in our model compared to having a stride of 1 where each pixel or node is examined, for example, and it is not so large that we lose a a lot of fidelity during training. 

The next portion of an encoder block is the convolutional batch normalization layer, which normalizes the output of the separable convolution layer described in the above paragraph. Using the mean and variance values of each mini-batch that a netowrk is trained on, we normalize the output of one layer before it is passed as an input to a new layer. Batch normalization allows us to train networks faster because it will converge more quickly despite each training iteration taking a little longer. It also allows us to use higher learning rates and adds noise to the network, preventing overfitting. 

I use three encoder layers to create a fairly deep network. These layers have 32, 64, and 128 filters to form a pyramidal encoder network. The smaller filters at the front of the network allow us to capture small details in an image, which are then collected and analyzed together using the larger filters in the following layers. This way we capture both high level and low level features in our network and extract more information from the image than if we only had one or the other. 

#### 1x1 Convolution

In order to avoid flattening the output shape of our encoder layers from a 4D tensor to a 3D tensor, we apply a 1x1 convolution layer to prevent the loss of spatial information. The 1x1 convolution allows seamless use of future convolutional layers, as a fully connected layer would lose the spatial information and reduce the efficacy of our upcoming convolutional decoder layers. Additionally, adding a 1x1 convolutional layer is cheaper than adding more regular convolution layers, whcih would require stacking larger and larger pyramid layers. 

#### Decoder blocks

The decoder blocks upsample our network from the 1x1 convolution back to the full sized image so that we can extract the spatial data about where our hero is and where the distractor people are. 

These blocks are composed of a bilinear upsampling layer, which expands an input shape using a 2x2 width and height. Bilinear upsampling linear interpolates the value of the R, G, and B color channels along the width and height axes of an input shape to fill in the RGB values that are missing after expanding the shape.

These bilinear upsampling layers and then followed by a concatenation layer, concatenating a previous decoder block output layer or 1x1 convolutional output layer with an encoder layer to better preserve spatial information. Since upsampling can remove some of the finer details from previous layers, concatenation allows us to preserve those details through the network. Concatenation has benefits over adding layers together because the depths of the layers do not have to be equal. The concatenation layer is followed by three separable convolution layers to extract more spatial information from the prior layers. 

I use three of these decoder blocks to mirror the number of encoder layers and expand the width and height of the output shape back to the original image size. This results in a reverse pyramid shape for the latter half of the network. The earlier decoder blocks are concatenated with later encoder blocks because the width and height of the layers must be equal. 

### Hyperparameters

The learning rate needs to be small enough to prevent the model from overshooting optimal model parameters when applying gradient descent during backpropagation, but it also needs to be large enough so that the model actually reaches the optimum in a reasonable amount of time. I played with the learning rate and observed the loss values for the training and validation sets until I found a well-paced learning rate of 0.01 with largely monotonic loss value reductions. 

The batch size determines how many data samples to train before calculating the mean and variance for the batch. A higher batch size can speed up the pace of optimization as it looks at multiple samples before moving down the gradient. A small mini-batch size could result in moving more obliquely down the gradient because the small sample misrepresents the direction of the gradient during backpropagation. I used a batch size of 32, which was a moderate size between 16 and 128. A batch size that is too large can slow down training time and converge slowly as a result. 

I used a relatively high epoch of 30 when training the initial network. An epoch is one forward pass and one backward pass of steps_per_epoch * batch_size. Using multiple epochs gives the network more opportunities to finetune its parameters on the same data, allowing the network to train better on the same data rather than getting more data. 

I used 100 steps per epoch, so each epoch has 100 * 32 samples. This value undershoots the total number of samples I have because I have a high number of epochs, allowing the network to sample from the total number of images to construct each epoch set randomly and reduce the possibility of overfitting. 

## Limitations

This trained model is only able to follow one "hero" human. It is unable to generalize to hero humans that don't look like the one in the training data.

The model also cannot generalize to do image segmentation on other entities besides humans, e.g. cars, animals, and other objects. 

A new model would have be trained with data labeled for a new hero or a new type of entity in order to address these limitations. 

## Future Improvements

Collecting more data could further improve the network. I relied on the existing data and manipulations of it, but collecting data that more directly addresses some of the false positives and negatives from the scoring could result in a better performing network. 

Writing a script to tune the hyperparameters on a small subset of the dataset could also result in more optimal hyperparameters compared to manual testing. 

## Resources Used

I referenced the Slack for learning how to flip images and increase the size of the default dataset. I also referenced Slack to learn how to output a diagram for my model using keras. 