
# Follow-Me Project
Congratulations on reaching the final project of the Robotics Nanodegree! 

Previously, you worked on the Semantic Segmentation lab where you built a deep learning network that locates a particular human target within an image. For this project, you will utilize what you implemented and learned from that lab and extend it to train a deep learning model that will allow a simulated quadcopter to follow around the person that it detects! 

Most of the code below is similar to the lab with some minor modifications. You can start with your existing solution, and modify and improve upon it to train the best possible model for this task.

You can click on any of the following to quickly jump to that part of this notebook:
1. [Data Collection](#data)
2. [FCN Layers](#fcn)
3. [Build the Model](#build)
4. [Training](#training)
5. [Prediction](#prediction)
6. [Evaluation](#evaluation)

## Data Collection<a id='data'></a>
We have provided you with a starting dataset for this project. Download instructions can be found in the README for this project's repo.
Alternatively, you can collect additional data of your own to improve your model. Check out the "Collecting Data" section in the Project Lesson in the Classroom for more details!


```python
import os
import glob
import sys
import tensorflow as tf

from scipy import misc
import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools
```

## FCN Layers <a id='fcn'></a>
In the Classroom, we discussed the different layers that constitute a fully convolutional network (FCN). The following code will introduce you to the functions that you need to build your semantic segmentation model.

### Separable Convolutions
The Encoder for your FCN will essentially require separable convolution layers, due to their advantages as explained in the classroom. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided below for your use. Each includes batch normalization with the ReLU activation function applied to the layers. 


```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

### Bilinear Upsampling
The following helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended, but you can try out different factors as well. Upsampling is used in the decoder block of the FCN.


```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

## Build the Model <a id='build'></a>
In the following cells, you will build an FCN to train a model to detect and locate the hero target within an image. The steps are:
- Create an `encoder_block`
- Create a `decoder_block`
- Build the FCN consisting of encoder block(s), a 1x1 convolution, and decoder block(s).  This step requires experimentation with different numbers of layers and filter sizes to build your model.

### Encoder Block
Create an encoder block that includes a separable convolution layer using the `separable_conv2d_batchnorm()` function. The `filters` parameter defines the size or depth of the output layer. For example, 32 or 64. 


```python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

### Decoder Block
The decoder block is comprised of three parts:
- A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
- A layer concatenation step. This step is similar to skip connections. You will concatenate the upsampled small_ip_layer and the large_ip_layer.
- Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.


```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    layer_1 = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    layer_2 = layers.concatenate([layer_1, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    layer_3 = separable_conv2d_batchnorm(layer_2, filters)
    layer_4 = separable_conv2d_batchnorm(layer_3, filters)
    output_layer = separable_conv2d_batchnorm(layer_4, filters)
    
    return output_layer
```

### Model

Now that you have the encoder and decoder blocks ready, go ahead and build your FCN architecture! 

There are three steps:
- Add encoder blocks to build the encoder layers. This is similar to how you added regular convolutional layers in your CNN lab.
- Add a 1x1 Convolution layer using the conv2d_batchnorm() function. Remember that 1x1 Convolutions require a kernel and stride of 1.
- Add decoder blocks for the decoder layers.


```python
def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc_1 = encoder_block(inputs, 32, 2)
    enc_2 = encoder_block(enc_1, 64, 2)
    enc_3 = encoder_block(enc_2, 128, 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    mid = conv2d_batchnorm(enc_3, 256, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dec_1 = decoder_block(mid, enc_2, 128)
    dec_2 = decoder_block(dec_1, enc_1, 64)
    dec_3 = decoder_block(dec_2, inputs, 32)
    x = dec_3
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

## Training <a id='training'></a>
The following cells will use the FCN you created and define an ouput layer based on the size of the processed image and the number of classes recognized. You will define the hyperparameters to compile and train your model.

Please Note: For this project, the helper code in `data_iterator.py` will resize the copter images to 160x160x3 to speed up training.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

# Call fcn_model()
output_layer = fcn_model(inputs, num_classes)
```

### Hyperparameters
Define and tune your hyperparameters.
- **batch_size**: number of training samples/images that get propagated through the network in a single pass.
- **num_epochs**: number of times the entire training dataset gets propagated through the network.
- **steps_per_epoch**: number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.
- **validation_steps**: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. We have provided you with a default value for this as well.
- **workers**: maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. We have provided a recommended value to work with. 


```python
learning_rate = 0.01
batch_size = 32
num_epochs = 15
steps_per_epoch = 100
validation_steps = 50
workers = 4
```


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Define the Keras model and compile it for training
# model = models.Model(inputs=inputs, outputs=output_layer)
weight_file_name = 'model_weights'
model = model_tools.load_network(weight_file_name)

model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

# Data iterators for loading the training and validation data
train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                               data_folder=os.path.join('..', 'data', 'train'),
                                               image_shape=image_shape,
                                               shift_aug=True)

val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                             data_folder=os.path.join('..', 'data', 'validation'),
                                             image_shape=image_shape)

logger_cb = plotting_tools.LoggerPlotter()
callbacks = [logger_cb]

model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)
```

    Epoch 1/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0211


![png](output_19_1.png)


    100/100 [==============================] - 86s - loss: 0.0212 - val_loss: 0.0289
    Epoch 2/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0215


![png](output_19_3.png)


    100/100 [==============================] - 85s - loss: 0.0214 - val_loss: 0.0683
    Epoch 3/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0198


![png](output_19_5.png)


    100/100 [==============================] - 84s - loss: 0.0198 - val_loss: 0.0295
    Epoch 4/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0195


![png](output_19_7.png)


    100/100 [==============================] - 85s - loss: 0.0195 - val_loss: 0.0641
    Epoch 5/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0178


![png](output_19_9.png)


    100/100 [==============================] - 85s - loss: 0.0179 - val_loss: 0.0323
    Epoch 6/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0174


![png](output_19_11.png)


    100/100 [==============================] - 84s - loss: 0.0173 - val_loss: 0.0290
    Epoch 7/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0186


![png](output_19_13.png)


    100/100 [==============================] - 85s - loss: 0.0185 - val_loss: 0.0359
    Epoch 8/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0204


![png](output_19_15.png)


    100/100 [==============================] - 85s - loss: 0.0204 - val_loss: 0.0659
    Epoch 9/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0178


![png](output_19_17.png)


    100/100 [==============================] - 85s - loss: 0.0178 - val_loss: 0.0279
    Epoch 10/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0161


![png](output_19_19.png)


    100/100 [==============================] - 85s - loss: 0.0161 - val_loss: 0.0203
    Epoch 11/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0172


![png](output_19_21.png)


    100/100 [==============================] - 84s - loss: 0.0172 - val_loss: 0.0334
    Epoch 12/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0164


![png](output_19_23.png)


    100/100 [==============================] - 85s - loss: 0.0164 - val_loss: 0.0232
    Epoch 13/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0172


![png](output_19_25.png)


    100/100 [==============================] - 84s - loss: 0.0171 - val_loss: 0.0446
    Epoch 14/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0147


![png](output_19_27.png)


    100/100 [==============================] - 85s - loss: 0.0148 - val_loss: 0.0247
    Epoch 15/15
     99/100 [============================>.] - ETA: 0s - loss: 0.0167


![png](output_19_29.png)


    100/100 [==============================] - 85s - loss: 0.0167 - val_loss: 0.0458





    <tensorflow.contrib.keras.python.keras.callbacks.History at 0x7f8ffc0f8160>




```python
# Save your trained model weights
weight_file_name = 'model_weights_flip'
model_tools.save_network(model, weight_file_name)
```

## Prediction <a id='prediction'></a>

Now that you have your model trained and saved, you can make predictions on your validation dataset. These predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well your model is doing under different conditions.

There are three different predictions available from the helper code provided:
- **patrol_with_targ**: Test how well the network can detect the hero from a distance.
- **patrol_non_targ**: Test how often the network makes a mistake and identifies the wrong person as the target.
- **following_images**: Test how well the network can identify the target while following them.


```python
# If you need to load a model which you previously trained you can uncomment the codeline that calls the function below.

# weight_file_name = 'model_weights'
# model = model_tools.load_network(weight_file_name)
```

The following cell will write predictions to files and return paths to the appropriate directories.
The `run_num` parameter is used to define or group all the data for a particular model run. You can change it for different runs. For example, 'run_1', 'run_2' etc.


```python
run_num = 'run_1'

val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_with_targ', 'sample_evaluation_data') 

val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                        run_num,'patrol_non_targ', 'sample_evaluation_data') 

val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                        run_num,'following_images', 'sample_evaluation_data')
```

Now lets look at your predictions, and compare them to the ground truth labels and original images.
Run each of the following cells to visualize some sample images from the predictions in the validation set.


```python
# images while following the target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','following_images', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
    
```


![png](output_26_0.png)



![png](output_26_1.png)



![png](output_26_2.png)



```python
# images while at patrol without target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_non_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
 
```


![png](output_27_0.png)



![png](output_27_1.png)



![png](output_27_2.png)



```python
   
# images while at patrol with target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_with_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
```


![png](output_28_0.png)



![png](output_28_1.png)



![png](output_28_2.png)


## Evaluation <a id='evaluation'></a>
Evaluate your model! The following cells include several different scores to help you evaluate your model under the different conditions discussed during the Prediction step. 


```python
# Scores for while the quad is following behind the target. 
true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)
```

    number of validation samples intersection over the union evaulated on 542
    average intersection over union for background is 0.993402508515974
    average intersection over union for other people is 0.36965377699331214
    average intersection over union for the hero is 0.905668035916945
    number true positives: 539, number false positives: 0, number false negatives: 0



```python
# Scores for images while the quad is on patrol and the target is not visable
true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)
```

    number of validation samples intersection over the union evaulated on 270
    average intersection over union for background is 0.9717286173845517
    average intersection over union for other people is 0.7208696798109584
    average intersection over union for the hero is 0.0
    number true positives: 0, number false positives: 102, number false negatives: 0



```python
# This score measures how well the neural network can detect the target from far away
true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)
```

    number of validation samples intersection over the union evaulated on 322
    average intersection over union for background is 0.9909734018234029
    average intersection over union for other people is 0.4262609232265483
    average intersection over union for the hero is 0.3611791188367818
    number true positives: 182, number false positives: 3, number false negatives: 119



```python
# Sum all the true positives, etc from the three datasets to get a weight for the score
true_pos = true_pos1 + true_pos2 + true_pos3
false_pos = false_pos1 + false_pos2 + false_pos3
false_neg = false_neg1 + false_neg2 + false_neg3

weight = true_pos/(true_pos+false_neg+false_pos)
print(weight)
```

    0.762962962962963



```python
# The IoU for the dataset that never includes the hero is excluded from grading
final_IoU = (iou1 + iou3)/2
print(final_IoU)
```

    0.633423577377



```python
# And the final grade score is 
final_score = final_IoU * weight
print(final_score)
```

    0.483278729406



```python
from tensorflow.contrib.keras.python.keras.utils import plot_model
plot_model(model, to_file='model.png')
```