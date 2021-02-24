# deep_learning_developer

This is a code repository built on top of PyTorch to speed up the development and testing of novel neural network models.

It is not targeted towards a user with little knowledge of deep learning. I developed this repository for my own research because I was tired of writing almost the same code with small differences for new models or to address new problems. Maybe another researcher will find this repository useful as well.

## How is this repository different from just using PyTorch?

This repository provides a general template for training neural networks, where the user fills in the blanks. PyTorch formalizes the components of deep learning and provides ready made classes and functions for using them i.e. loss functions, optimizers, layers for composing neural networks. However, when using PyTorch, if the user wants to develop novel neural network models and training objectives, it is up to them to write the code to stitch the components together. This repository provides the stitching and instead asks the user fill out a configuration file where they can provide details such as which model to use, which loss function to train the model with and which evaluation metrics to use to track the training.

In addition, logging training results is integrated into the training process with a Logger class and by providing wrappers for evaluation and loss functions. Saving and loading models is also integrated into the training and testing process by providing wrappers for nn.Modules and for models composed of multiple nn.Modules.

This repository also provides a number of heuristics to define the architectures of specific neural network types like convolutional neural networks. So generating a new neural network module can be done by specifying the required input and output sizes of the tensors being fed through the network. This means that a user does not have to define the parameters for each convolutional layer in the network which can be a tedious process.

The software creates a development data set when generating the training and validation set for training a model. The development set is very small and can be used for running a model on a CPU alone or for debugging code.

## Installation Instructions

1. Install anaconda on your computer - https://docs.anaconda.com/anaconda/install/

2. Open a terminal and create a new anaconda environment with python 3.7+
```bash
conda create -n deeplearning python=3.7
conda activate deeplearning
```

3. Check your CUDA version
```bash
nvcc --version
```
4. Install repository dependencies. For installing pytorch, check https://pytorch.org/ and match the install command with your CUDA version. The command below is for a computer with CUDA version 10 installed.
```bash
conda install matplotlib
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorboard
conda install -c conda-forge tensorboardX
conda install -c intel scikit-learn
pip install pyyaml
```

5. Install `deeplearning` repository
```bash
mkdir <PATH WHERE YOU WANT TO STORE THE REPOSITORY>
cd <PATH WHERE YOU WANT TO STORE THE REPOSITORY>
git clone https://github.com/zachares/deep_learning.git
cd deep_learning
pip install -e .
```
6. Test your installation
```bash
cd example_testing
python example_testing.py
```
`example_testing.py` should run without issue.
```bash
cd ../example_training
python example_training.py
```
`example_training.py` should run without issue (but requires user input during the run)

## Important!
Whenever you want to run programs from this repository, first you have to activate the anaconda environment you created using the command in terminal:
```bash
conda activate deeplearning
```
You can deactivate the environment when you are done using the command:
```bash
conda deactivate
```
## Getting started

After installation, you can refer to the example project to see how the code is used (in folders `example_testing` and `example_training`). The example project is to classify images from the CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html for a reference on the data set and the researchers who developed it). The example project uses a convolutional neural network with a set of ResNet layers at the end of it to classify images. There is a trained model in the example_testing folder which can be used as a baseline (it doesn't work very well). The files in the example project are meant to act as a template for a new project. Below I describe how to develop components for the example project, but this process is the same for a new project as well.

### Defining a new model

Models are defined in `example_training/example_models.py`. To define a new model, copy and paste below (in the same file) the example model class `Cifar10Classifier`. There are three class methods that need to be modified as well as its name:

Methods that need to be modified 

1. `__init__()`: specifically, the initialization arguments i.e. `self.num_resnet_layers` and the neural network modules of the model i.e. `self.image_processor, self.encoding_classifier`. The initialization arguments need to be modified according to the design of your new model, i.e. you may want a different output size for the `image_processor`. The neural network modules need to be modified according to the architecture of your new model. The file `models_modules.py` contains the modules that can be used to compose a new model, please see this file for documentation on the modules. Currently, the file contains definitions for modules composed of: 2D convolutional neural networks, 2D 'deconvolutional' neural networks, 1D convolutional neural networks, 1D 'deconvolutional' neural networks, embedding layers, fully-connected networks, ResNet-esque networks, transformer networks and transformer decoder networks (called TransformerComparer, because I have used it in the past to compare a time series to itself in my research). Note, I put 'deconvolutional' in brackets because the layers are not performing a deconvolution operation, but are commonly referred to as deconvolutional layers.

2. `forward()`: if your new model uses additional inputs and/or processes its inputs differently than the current model class.

3. `classify()`: if you need to preprocess test inputs differently than the `Cifar10Classifier` does. This method can have any name (you may want to change it if you the model estimates or predicts a real valued vector instead of performing classification). What this method should do is process test inputs and then feed them through the network to perform classification, estimation or prediction at test time.

After the new class has been defined, it needs to be added to the dictionary in the function `get_ref_model_dict` in the same file `example_training/example_models.py`. In the dictionary, the key for the class should be its name as a string. This step is important so that the model can be initialized and loaded using a config file.

### Defining a new dataset

The data set / data loader for the example project is in `example_training/dataloader.py`. The current software does not support using multiple data sets for a single project, because it hasn't come up in my research (and I believe that you shouldn't add functionality to software unless necessary). So, if you are defining a new data set, you can modify the `CustomDataSet` class instead of creating a new class like we did when defining a new model. To define a new data set, you need to modify two methods of the class `__init__()` and `__getitem__()`. 

A key component of the way I have designed the `CustomDataSet` as a generalizable class is the use of the attribute / dictionary `idx_dict`. The `__getitem__()` method of the `CustomDataSet` takes an index as an argument which is used to load a single data point from a data set. The `idx_dict` attribute is a dictionary which stores a mapping from that index to where the data point is stored. In a past project data points were stored in individual h5 files and the `idx_dict` mapping was from an index to a string containing the path to an h5 file. For the example project, we load the data set into memory as a np.ndarray in the `__init__` method and add the data set as attributes to the `CustomDataSet` class (`self.cifar_train_data`, `self.cifar_train_labels`). So for the example project `idx_dict` is a mapping from one index to another. The importance of the `idx_dict` is that it stores the seperation of the full data set into a validation, training and development set while providing a mapping for loading data points from each set. I use this method, because it means you don't not to define a seperate `CustomDataSet` class for the training, validation and development set. 

Unfortunately, I cannot provide a set of step by step instructions for how to modify the `CustomDataSet` class to a new project. In the `__init__()` method, if the data set used for training is small enough that it can be loaded into memory, then the user should modify the method to load the data set. If not, then the user should create a list with a reference to each file where the data sets are contained. For each data point, append the reference to that point to the list `self.idx_dict['idx_list']`. Then the code below where `self.idx_dict['idx_list']` is defined will automatically split up the data set into a validation, training and development set.

You want to modify the `__getitem__()` method, so that it loads all the inputs from your data set required for training into a dictionary of torch.Tensors.

### Defining a new training run

Configuration files are used to define a training run. This is useful because if you have your data set and models already defined then training a model is as simple as editing a yml file instead of having to write out an entire program for training, logging and saving models. Unfortunately, training a neural network is a relatively complicated process, so the configuration file has a lot of parameters stored in it. Below, I added the annotated config file `example_learning_config.yml ` which has inline comments on the use of each parameter / component in the config file. I know there are a lot of parameters, but for the most part you will be keeping the parameters the same at the beginning of a new project. The main things that will need to be changed at the beginning of a project are `dataset_path`, `logging_dir` and the dictionary `info_flow`.

```yml 
# the parameters required to define a data loader
dataloading_params:
  # the path to where the data set for the project is stored
  dataset_path: "../cifar_10_dataset" 
  # the fraction of the data set to use in the validation set
  val_ratio: 0.05
  # the approximate number of data points that should be in the development set
  dev_num: 20
  # the number of CPU cores to use for loading data
  num_workers: 4
  # the size of the random batches used to train a neural network
  batch_size: 128
  # if you want to use the same decomposition of the data set into validation, training
  # and development sets from a previous run, write the path to the previous idx_dict here,
  # each time a model is trained and results are logged and, or models are saved, the 
  # idx_dict used for training is automatically saved in the same folder
  idx_dict_path: 

# the parameters required for training a model
training_params:
  # the random seed used for the run, this is specified to make the training process more 
  # repeatable
  seed: 4321
  # All the parameters until the next comment are used as an input to PyTorch's version 
  # of the ADAM optimizer (please see https://pytorch.org/docs/stable/optim.html for an 
  # explaination of the parameters)
  regularization_weight: 0.00001
  lrn_rate: 0.0001
  beta1: 0.9
  beta2: 0.99
  # the maximum number of epochs to train a model for
  max_training_epochs: 10000
  # whether to use the development set for the run (if you want to debug your code)
  use_dev: False
  # whether to use a GPU to train the model, (requires that your computer has a GPU and
  # has CUDA or some other software installed to interface the GPU with PyTorch code
  use_GPU: True

# the parameters required for logging the results of training and saving models
logging_params:
  # The path to the directory where training results and models will be saved
  # Place your logging dir in a seperate location then your local code repository
  logging_dir: /home/ubuntu/src/example_logging/
  # a small additional note which is added to the logging directory, I find it useful 
  # if I am traininga lot of models in a short period of time which are slightly different
  run_notes: system_test

# the info_flow dictionary is the key to using this repository. All the above parameters 
# can be essentially left as they are (except the data set loading and logging paths) for a 
# new project until you get to the hyperparameter tuning phase. 'info_flow' defines which 
# models will be trained and how. Each key in info_flow should be the name of the class of 
# a model you want to load and,or train.
info_flow:
  # a model that you want to load or train
  Cifar10Classifier:
    # whether you want to train the model (sometimes you want to load a previously trained 
    # model, but only to get its outputs instead of to train it) 1 = train, otherwise the model 
    # will not be trained
    train: 1
    # the path to the directory where the weights of a trained model are stored (the 'logging_dir' 
    # from a previous run)
    model_dir:
    # the epoch during the previous training sequence during which the model was saved
    epoch: 0
    # a dictionary with the initialization arguments for the model
    init_args:
      image_size: [3,32,32]
      encoding_size: [64,1,1]
      num_resnet_layers: 3
    # a dictionary with the inputs to the model required for a forward pass and their origin 
    # (i.e. cifar-10 images 'image' are loaded from the data set)
    inputs:
      image: dataset
    # a dictionary with the loss functions used to train the model  
    losses:
      # the name of the loss function in the loss dict (see utils_sl.py)
      MultinomialNLL:
        # the name to use for the loss when logging
        logging_name: Cross_Entropy_loss
        # the weight to multiply the loss value by for training
        weight: 1.0
        # the inputs required for the loss function
        inputs:
          class_logits: Cifar10Classifier
          class_idx: dataset
    # a dictionary of the evaluation metric functions used to evaluate the training 
    # of the model
    evals:
      # the name of the evaluation metric in the eval dict (see utils_sl.py)
      MultinomialAccuracy:
        # the name for the evaluation metric when logging
        logging_name: "Cifar-10 Accuracy"
        # the inputs required for the evaluation metric function
        inputs:
          class_logits: Cifar10Classifier
          class_idx: dataset
```

Once the configuration file is correctly set up, to train a model run: 
```bash
conda activate deeplearning
python example_training.py
```
### Viewing the training results of a run

The training results can be viewed any time during training or after. This repository uses tensorboard to log results. When you run:
```bash
python example_training.py
```
and specify that you want to log results or save models. The program prints out a line to terminal at the beginning of training which looks like:
```bash
Logging and Model Saving Directory: <PATH TO DIRECTORY WHERE RESULTS AND MODELS ARE SAVED>
```
To view the results, copy the printed out path and open a new terminal. Then run the commands:
```bash
conda activate deeplearning
cd <PATH TO DIRECTORY WHERE RESULTS AND MODELS ARE SAVED>
tensorboard --logdir=.
```
The argument `--logdir=` specifies the directory to look for results in and the `.` means the current directory. So you could also just use the command
`tensorboard --logdir= <PATH TO DIRECTORY WHERE RESULTS AND MODELS ARE SAVED>`

### Adding custom loss functions, evaluation metric functions and neural network modules

There is a file called `example_training/training_utils.py` where new loss functions and evaluation metric functions can be defined in the function `get_project_loss_and_eval_dict()`. In addition, you can add project-specific neural network module classes if you want. As an example of defining a custom loss function see how `IdentityLoss` is defined in the loss dictionary in the function `get_loss_and_eval_dict()` in `utils_sl.py`. As an example of a custom evaluation metric function see how `ContinuousAccuracy` is defined in the eval dictionary in the function `get_loss_and_eval_dict()` in `utils_sl.py`. As an example of a custom neural network module see how the class `CONV2DN` is defined in `models_models.py`.






