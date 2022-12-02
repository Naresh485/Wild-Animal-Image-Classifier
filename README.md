# Wild-Animal-Image-Classifier

The object oif this blog is to classify images from a multiple class images.

Image Classification :
Image classification refers to the task of extracting information classes from a multiband raster image. The task of classifying and labeling groups of pixels or vectors inside an image based on certain rules is known as image classification. A model is trained to identify various image classes. A CNN model, for example, maybe trained to detect images of multiple different types of animals: tiger, cheetah, lion, wolf etc.
Depending on the interaction between the analyst and the computer during classification, there are two types of classification: supervised and unsupervised.

Convolutional Neural Networks:
In last few decades, deep learning has proved to be very important and powerful tool because of its capacity  to deal with large datasets.



Convolutional Neural Networks are most commonly used for image classification. In early 1980 CNN can be  used to detect handwritten words.CNN uses convolution instead of matrix multiplication. Convolution in mathematics is mathematical operation on two functions that produce a third function which express the shape of one is modified by the other.


Neural Networks:

Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.
Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.


Reference: Source 

Building blocks of CNN:
Convultional neural networks outperforms other neural networks when handling with images, speech or audio signal inputs.
A CNN has the following layers.
Input layers:
This is the layer where input image size to the model is specified.
Activation Function:

The outcome of the neural network is determined by mathematics. This function is applied to each neuron in the network and determines whether or not the neuron should be activated based on the input relevant to the model prediction.
There are different types of activation functions they are ReLu, SoftMax, LeakyReLu etc.

Pooling layer
The pooling layer is used to reduce the number of parameters and computations in the network gradually decreasing the spatial size of the representation. It also aids in the prevention of overfitting.
Pooling methods come in a variety of shapes and sizes. The most crucial are listed below.
1.Max pooling : It is a convolution procedure in which the Kernel or feature detector recovers the maximum value of the area that it convolves.
2.Min pooling: In min pooling, the kernel extracts the minimum value of the area it convolves in the same way it extracts the minimum value of the area it convolves.
3.Average pooling: In average pooling, the kernel extracts the area’s average value.
In our models, we are using max-pooling and average pooling operation. Max pooling is also called downsampling.

filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
Arguments
filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
padding: one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input.
data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.
dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
groups: A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.
activation: Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).
use_bias: Boolean, whether the layer uses a bias vector.
kernel_initializer: Initializer for the kernel weights matrix (see keras.initializers). Defaults to 'glorot_uniform'.
bias_initializer: Initializer for the bias vector (see keras.initializers). Defaults to 'zeros'.
kernel_regularizer: Regularizer function applied to the kernel weights matrix (see keras.regularizers).
bias_regularizer: Regularizer function applied to the bias vector (see keras.regularizers).
activity_regularizer: Regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).
kernel_constraint: Constraint function applied to the kernel matrix (see keras.constraints).
bias_constraint: Constraint function applied to the bias vector (see keras.constraints).
Input shape
4+D tensor with shape: batch_shape + (channels, rows, cols) if data_format='channels_first' or 4+D tensor with shape: batch_shape + (rows, cols, channels) if data_format='channels_last'.
Output shape
4+D tensor with shape: batch_shape + (filters, new_rows, new_cols) if data_format='channels_first' or 4+D tensor with shape: batch_shape + (new_rows, new_cols, filters) if data_format='channels_last'. rows and cols values might have changed due to padding.

Code Analysis:

Importing libraries required for the model building.


Pre- Processing the data : Input is read from the kaggle dataset. The ImageDataGenerator class contains three methods: flow(), flow from directory(), and flow from dataframe(). In this we  are using flow_from_directory for this model.
 The directory must be set to the path where ’n’ classes of folders are present.
The target_size is the size of your input images, every image will be resized to this size.
batch_size: No. of images to be yielded from the generator per batch.
class_mode: Set “binary” if you have only two classes to predict, if not set to“categorical”, in case if you’re developing an Auto encoder system, both input and the output would probably be the same image, for this case set to “input”.
shuffle: Set True if you want to shuffle the order of the image that is being yielded, else set False.
seed: Random seed for applying random image augmentation and shuffling the order of the image.

Image Augmentaion:

ImageDatagenerator class  provides a easy and quick way to augment our images.It provides a host of different augmentation techniques like standardization, rotation, shifts, flips, brightness change, and many more.

Data Splitting:
In this model we are splitting data as train data as 80% and Test data as 20%.

Model1 :

CNN model1 is implemented with 3 convolutional layers with 64,32,128 filters and kernel size 3,3,2 respectively. Averagepooling of pool size(1,1).



Output:


Model fitting:

We are fitiing the model with train and validation  datasets sonsidering epoch as 5.



 
The accuracy of the first model is 6.67 . Plotting model accuracy and model graph.




Model2:

Model 2 is implemented with three different Conv2D layers with 128,64,48 filter and (2,2) as kernal size respectively, by taking MaxPooling2D of pool_size (3,3). and activation function as LeakyReLu.




Model fitting:

We are fitiing the model with train and validation  datasets sonsidering epoch as 5.



The accuracy of the Model2 is 67% , plotting model accuracy and model graph.


 We are using model.save() to save the model.
We are randomly taking 4 images and predicting them  using the model.






Challenges and Solution:
Identifying the right parameters to increase the accuracy of the model was challenging.
Reducing overfitting was really challenging and solved this problem by referring to available model on the internet and applying techniques like dropout layer and data augmentation.
Contribution:
I have used Maxpooling for model 1 and Averagepooling for the second model and founf the change in performance.
I have developed this model by referring to the available model and made several experiments by changing the parameters like activation function, kernel size and filters of the CNN model to increase the accuracy of the model.
Created multiple layers to check the performance, for every model i have increased conv2d layers to check the perfomance of the model.
Change the hyperparameters- i.e neurons per layer
experimented on epoch values for 3 models 5,5, 28.
Used diffferent activation functions like relu, LeakyRelu, softmax.



https://www.youtube.com/watch?v=0dI8_kP-xSM&ab_channel=NareshKandhyanam 

References:
https://keras.io/api/layers/convolution_layers/convolution2d/ 
https://medium.com/@likithaveluri1998/natural-images-classification-from-scratch-e120f05691ba
https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/  
https://www.ibm.com/cloud/learn/neural-networks 
https://github.com/likithaveluri/natural_images_classifiers/blob/main/DM_Final_1002025528.ipynb 
