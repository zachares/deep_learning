# deep_learning_developer

This is a code repository built on top of PyTorch to speed up the development and testing of novel neural network models.

It is not targeted towards a user with little knowledge of deep learning. I developed this repository for my own research because I was tired of writing almost the same code with small differences for new models or to address new problems. Maybe another researcher will find this repository useful as well.

## How is this repository different from just using PyTorch?

This repository provides a general template for training neural networks, where the user fills in the blanks. PyTorch does a good job of seperating the components of deep learning, i.e. loss functions, optimizers, layers for composing neural networks. However, when using PyTorch, if the user wants to develop novel neural network models and training objectives, it is up to them to write the code to stitch the components together. This repository provides the stitching and instead asks the user fill out a configuration file where they can provide details such as which model to use, which loss function to train the model with and which evaluation metrics to use to track the training.

In addition, logging training results is integrated into the training process with a Logger class and by providing wrappers for evaluation and loss functions. Saving and loading models is also integrated into the training and testing process by providing wrappers for nn.Modules and models composed of multiple nn.Modules.

This repository also provides a number of heuristics to define the architecture of specific neural network types like convolutional neural networks so generating a new neural network module can be done by specifying the required input and output sizes of the tensors being fed through the network. This means that a user does not have to define the parameters for each convolutional layer in the network which can be a tedious process.

## Getting started

If you are interested in using this repository for your own development. Please follow the instructions in the INSTALLATION.md markdown to install the package and its dependencies. I prefer this method of installation to using a requirements.txt file because those tend to contain a lot of packages which are not necessary to running the code and specify a specific version for each dependency even though it is not required.

After installation, you can refer to the example project to see how the code is used. The example project is to classify images from the CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html for a reference on the data set and the researchers who developed it). The example project uses a convolutional neural network with a set of ResNet layers at the end of it. There is a trained model in the example_testing folder which can be used as a baseline. It doesn't perform very well, but by running ```python example_testing.py``` you can test to make sure everything is working and installed correctly. If you want to train a model go to the example_training folder and run ```python example_training.py```. In both folder there is a config_file which is used to specify the parameters for the run. A markdown about config files will be added shortly.






