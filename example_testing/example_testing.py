import sys
import os
import yaml
import random
import numpy as np
import utils_sl as sl

sys.path.append("../example_training")
sys.path.append("../cifar_10_dataset")

from load_cifar_10 import load_cifar_10_data

import example_models as em

import multinomial as multinomial
import torch
import matplotlib

def testing_example():
    """ Tests a network's ability to classify images in the CIFAR-10
        test data set and prints out the test results
    """
    # Loads the configuration file for this testing program
    with open("example_testing_config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Setting the random seed to make the results repeatable if, there is
    # sampling in the evaluation
    seed = cfg['testing_params']['seed']
    random.seed(seed)
    np.random.seed(seed)

    # Checks whether a GPU is available for training and whether the
    # configuration file specifies to use a GPU 
    use_cuda = cfg["testing_params"]["use_GPU"] and torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    # Loads a reference to the specified hardware (CPU or GPU)
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Loads the neural network architectures from project_models.py 
    # specified in learning_params.yml into a dictionary 
    model_dict = sl.init_and_load_models(em.get_ref_model_dict(),
                                         cfg['info_flow'],
                                         device)
    
    assert len(model_dict.keys()) == 1, ("More than one model has been "
                                         + "selected for testing, but " 
                                         + "the testing program can " 
                                         + "only test one model per run")
    
    test_model = model_dict[list(model_dict.keys())[0]]

    # Loading test set
    data_dir = "../cifar_10_dataset"
    (cifar_train_data, 
     cifar_train_filenames,
     cifar_train_labels,
     cifar_test_data, 
     cifar_test_filenames, 
     cifar_test_labels, 
     cifar_label_names) = load_cifar_10_data(data_dir)

    batch_size = 500

    # Calculating Training Set Accuracy
    class_acc = None
    for i in range(int(np.ceil(cifar_train_data.shape[0] / batch_size))):
        if i > cifar_train_data.shape[0] / batch_size:
            idx0 = i * batch_size
            idx1 = cifar_train_data.shape[0]
        else:
            idx0 = i * batch_size
            idx1 = (i+1) * batch_size
        
        batch_data = cifar_train_data[idx0:idx1]
        
        batch_logits = test_model.classify(batch_data)
        
        batch_labels = torch.Tensor(cifar_train_labels[idx0:idx1]).long().to(batch_logits.device)

        class_acc_batch = multinomial.logits2acc(batch_logits, batch_labels)

        if class_acc is None:
            class_acc = class_acc_batch.clone()
        else:
            class_acc = torch.cat([class_acc, class_acc_batch], dim = 0)
    
    print("The classification accuracy of your model was " +
          "{} on the CIFAR-10 train set.".format(class_acc.mean().item()))

    # Calculating Test Set Accuracy
    class_acc = None
    for i in range(int(np.ceil(cifar_test_data.shape[0] / batch_size))):
        if i > cifar_test_data.shape[0] / batch_size:
            idx0 = i * batch_size
            idx1 = cifar_test_data.shape[0]
        else:
            idx0 = i * batch_size
            idx1 = (i+1) * batch_size
        
        batch_data = cifar_test_data[idx0:idx1]
        
        batch_logits = test_model.classify(batch_data)

        batch_labels = torch.Tensor(cifar_test_labels[idx0:idx1]).long().to(batch_logits.device)

        class_acc_batch = multinomial.logits2acc(batch_logits, batch_labels)

        if class_acc is None:
            class_acc = class_acc_batch.clone()
        else:
            class_acc = torch.cat([class_acc, class_acc_batch], dim = 0)
    
    print("The classification accuracy of your model was " +
          "{} on the CIFAR-10 train set.".format(class_acc.mean().item()))

    # Printing a histogram of the accuracy of the model per class on the 
    # test set
    probs = []
    for i in range(10):
        probs.append(class_acc[cifar_test_labels == i].mean().item())

    multinomial.print_histogram(torch.Tensor(probs),
                                cifar_label_names,
                                histogram_height=10)

if __name__ == "__main__":
	testing_example()




