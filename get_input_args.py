#
# Functions for getting input arguments
# Created by: AHMET CELIK
#

import argparse

def get_train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", type = str, help = "path to the folder of images")
    parser.add_argument("--save_dir", type = str, default = "checkpoint_part2.pth", help = "checkpoint directory to save model")
    parser.add_argument("--arch", type = str, default = "vgg16", help = "neural network model (vgg16 - alexnet - densenet121)")
    parser.add_argument("--learning_rate", type = float, default = 0.001, help = "learning rate")
    parser.add_argument("--epochs", type = int, default = 5, help = "number of epochs")
    parser.add_argument("--hidden_units", type = int, default = 1024, help = "number of units in the first hidden layer")
    parser.add_argument("--gpu", action="store_true", help = "enable gpu")
    
    return parser.parse_args()

def get_predict_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("path_to_image", type = str, help = "image path for prediction")
    parser.add_argument("checkpoint", type = str, help = "checkpoint directory of model")
    parser.add_argument("--top_k", type = int, default = 3, help = "return top K results")
    parser.add_argument("--category_names", type = str, default = "cat_to_name.json", help = "mapping of categories to names")
    parser.add_argument("--gpu", action="store_true", help = "enable gpu")
    
    return parser.parse_args()