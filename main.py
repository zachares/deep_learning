import os
import sys
import torch
import yaml
import random
import numpy as np
import time

from trainer import Trainer
from logger import Logger
from supervised_learning_utils import init_dataloader

######## git hash to know the state of the code when experiments are run
### add saving train log

def main():
    ##################################################################################
    ##### Loading required config files
    ##################################################################################
    with open("learning_params.yml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    use_cuda = cfg['training_params']['use_GPU'] and torch.cuda.is_available()
    seed = cfg['training_params']['seed']
    run_mode = cfg['training_params']['run_mode'] 

    idx_dict_path = cfg['dataloading_params']['idx_dict_path']

    save_val_interval = cfg['logging_params']['save_val_interval']

    test_run = False

    if run_mode == 0:
        run_description = "development"
        cfg['training_params']['val_ratio'] = 0
    elif run_mode == 1:
        run_description = "training"
    elif run_mode == 2:
        run_description = "testing"
        cfg['training_params']['val_ratio'] = 0
        cfg['training_params']['max_training_epochs'] = 0
        test_run = True
    else:
        raise Exception("Sorry, run mode " + str(run_mode) + " is not supported")

    val_ratio = cfg['training_params']['val_ratio']
    max_epoch = cfg['training_params']['max_training_epochs'] + 1

    print("Running code in " + run_description + " mode")

    if idx_dict_path == "":
        idx_dict_path = None

    ##################################################################################
    ### Setting Debugging Flag and Save Model Flag
    ##################################################################################
    var = input("Run code in debugging mode? If yes, no Results will be saved.[yes,no]: ")
    if var == "yes":
        debugging_flag = True
    elif var == "no":
        debugging_flag = False
    else:
        raise Exception("Sorry, " + var + " is not a valid input for determine whether to run in debugging mode")

    if debugging_flag:
        print("Currently Debugging")
        torch.autograd.set_detect_anomaly(True)
    else:
        print("Training with debugged code")

    var = input("Run code without saving models?[yes,no]: ")
    if var == "yes":
        save_model_flag = False
    elif var == "no":
        save_model_flag = True
    else:
        raise Exception("Sorry, " + var + " is not a valid input for determine whether to save models" )
    ##################################################################################
    # hardware and low level training details
    ##################################################################################
    device = torch.device("cuda" if use_cuda else "cpu")
    random.seed(seed)
    np.random.seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    if use_cuda:
      print("Let's use", torch.cuda.device_count(), "GPUs!")

    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
    logger = Logger(cfg, debugging_flag, save_model_flag, run_description)

    ##################################################################################
    #### Training tool to train and evaluate neural networks
    ##################################################################################
    trainer = Trainer(cfg, logger.models_folder, save_model_flag, device)

    ##################################################################################
    #### Dataset creation function
    ##################################################################################
    data_loader, val_data_loader, idx_dict = init_dataloader(cfg, device, idx_dict_path)

    if save_model_flag:
        logger.save_dict("val_train_split", idx_dict, False)
        trainer.save(0)
    ##################################################################################
    ####### Training ########
    ##################################################################################
    global_cnt = 0
    i_epoch = 0
    val_global_cnt = 0
    prev_time = time.time()

    if not test_run:
        for i_epoch in range(max_epoch):
            current_time = time.time()

            if i_epoch != 0:
                print("Epoch took ", current_time - prev_time, " seconds")
                prev_time = time.time()

            print('Training epoch #{}...'.format(i_epoch))
            
            for i_iter, sample_batched in enumerate(data_loader):
                sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
                sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()

                logging_dict = trainer.train(sample_batched)
                global_cnt += 1

                if global_cnt % 50 == 0 or global_cnt == 1:
                    print(global_cnt, " Updates to the model have been performed ")
                    logger.save_images2D(logging_dict, global_cnt, 'train/')

                logger.save_scalars(logging_dict, global_cnt, 'train/')
            ##################################################################################
            ##### Validation #########
            ##################################################################################
            # performed at the end of each epoch
            if val_ratio is not 0 and (i_epoch % save_val_interval) == 0:
                
                print("Calculating validation results after #{} epochs".format(i_epoch))

                for i_iter, sample_batched in enumerate(val_data_loader):
                    sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
                    sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()

                    logging_dict= trainer.eval(sample_batched)

                    logger.save_scalars(logging_dict, val_global_cnt, 'val/')

                    val_global_cnt += 1

                logger.save_images2D(logging_dict, val_global_cnt, 'val/')

            ###############################################
            ##### Saving models every epoch ################
            ##############################################
            if save_model_flag and i_epoch == 0:
                if os.path.isdir(logger.models_folder) == False:
                    os.mkdir(logger.models_folder)
                trainer.save(i_epoch)

            elif save_model_flag and (i_epoch % save_val_interval) == 0:
                trainer.save(i_epoch)

    else:
        print("Starting Testing")

        for i_iter, sample_batched in enumerate(data_loader):
            sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
            sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()

            logging_dict= trainer.eval(sample_batched)

            logger.save_scalars(logging_dict, global_cnt, 'test/')

            if (global_cnt + 1) % 50 == 0 or (global_cnt + 1) == 1:
                print(global_cnt + 1, " samples tested")

            global_cnt += 1

        logger.save_images2D(logging_dict, global_cnt, 'test/')

        print("Finished Testing")

if __name__ == '__main__':
    main()