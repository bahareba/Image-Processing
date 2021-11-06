# About Project:
its my first time to working with tensorflow and specially neural network. in this project  I'm gonna train a neural network to detect a different range of clothes. those are classify in several groups e.g T-shirt,shoes, ... and I'm gonna classify them into 10 groups using neural network. I will use tf.keras too, because its a high-level API to build and train models in tensorflow.
so lets get started!
# Import data:
in this project I use a dateset which include 70.000 images from 10 category clothes. we called it Fashion-MNIST dataset. Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We can access the Fashion MNIST data set directly from TensorFlow. Import and load the Fashion MNIST data directly from TensorFlow.
As you know, to start a neural network  we should dived data set into four NumPy arrays. two of them are related to training set consist of train_image and train_labels and two of them are related to test set which consist of test_image and test_labels. 
as I mentioned, the images are 28x28 NumPy arrays and pixel values ranging from 0 to 255. The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
Label | Class
------------ | -------------
0 | T-shirt/top
1 | Trouser
2 | Pullover
3 | Dress
4 | Coat
5 | Sandal
6 | Shirt
7 | Sneaker
8 | Bag
9 | Ankle boot


Each image is blogs to a single label. Since there is not class name column in data set, we should store them to use later when plotting the images.

# EDA
Let's explore the format of the dataset before training the model. The training set shows there are 60,000 images , with each image represented as 28 x 28 pixels Likewise, there are 60,000 labels in there. Each label is an integer between 0 and 9 because as I mentioned i they correspond to the class of clothing the image represents.verify that the data is in the correct format and that you're ready to build and train the network, let's display the first 25 images from the training set and display the class name below each image.
There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels and it contains 10,000 images labels.

# Preprocess
before starting preprocess lets inspect the first image in the training set, as you see the pixel values  fall in the range of 0 to 255.
in the next step we need to scale these values to range of 0 to 1 before feeding them to the neural network. To do this we should divide all of the by 255 both in training and test set.
To make sure that we did it correctly lets display first 25 images from training set  with their class name using for loop.

# Modeling
Building neural network model requires configuration the layers of the model, so we should take two step: First  set up the layers and then compiling the model.
## First step:
The basic building block of a neural network is the layer. Layers extract representations from the data fed into them.Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training. lets explain our layers in this neural network:
1. The first layer is Flatten, It has not any parameters to learn, it just reformat the data. Each image has two dimension (28*28 pixel). In this layer it transform from two-dimension array to one-dimension array.
2. After that, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). 
3. The last layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes
## Second step:
In this step we need a few more setting before training the model.
1. Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction. loss function measures error and the lower loss function is better, in the other hand in deep learning model our goal is minimized the loss function.
2. Optimizer —This is how the model is updated based on the data it sees and its loss function.
3. Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.


# Train the model
Training the neural network model requires the following steps:
1. Using model.fir feed the training data to the model. In this example, the training data is in the train_imagesand train_labelsarrays. As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.85 (or 85%) on the training data.
2. The model learns to associate images and labels.
3. You ask the model to make predictions about a test set—in this example, the test_imagesarray.
4. Verify that the predictions match the labels from the test_labelsarray.


# Evaluate
In the next step,we compare how the model performs on the test data set.It turns out that the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy represents overfitting.


# Make predictions
With the model trained, we can use it to make predictions about some images. The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.The Logit function is used similarly to the sigmoid function in neural networks. The sigmoid, or activation, function produces a probability, whereas the Logit function takes a probability and produces a real number between negative and positive infinity. Like the sigmoid function, Logit functions are often placed as the last layer in a neural network as can simplify the data. For example, a Logit function is often used in the final layer of a neural network used in classification tasks. As the network determines probabilities for classification, the Logit function can transform those probabilities to real numbers. 
then  the model has predicted the label for each image in the testing set. Let's take a look at the 6th prediction.
A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. we can see which label has the highest confidence value, so as you see the highest confidence is belongs to 4th label and is is same in test labels.
To look at the full set of 10 class predictions lets graph this.


# Verify predictions

With the model trained, we can use it to make predictions about some images. Let's look at the 10th image, predictions, and prediction array. Correct prediction labels are green and incorrect prediction labels are red. The number gives the percentage (out of 100) for the predicted label.