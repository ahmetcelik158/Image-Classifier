#
# TRAIN.PY
# Created by: AHMET CELIK
#

import ai_methods
from get_input_args import get_train_args

arch_inp_size = {"vgg16": 25088,
                 "alexnet": 9216,
                 "densenet121": 1024}

def main():
    train_args = get_train_args()
    current_arch = "vgg16" # CHECKING IF VALID ARCH IS ENTERED. OR USE VGG16 
    if train_args.arch in arch_inp_size:
        current_arch = train_args.arch
    my_device = "cuda" if train_args.gpu else "cpu"
    
    dataloaders, class_to_idx = ai_methods.get_loader_and_classidx(train_args.data_dir)
    input_size = arch_inp_size[current_arch]
    output_size = len(class_to_idx)
    hidden1 = train_args.hidden_units # creating 2 hidden layers
    hidden2 = max(int(input_size//4), output_size*4)
    hidden_layers = sorted([hidden1, hidden2], reverse=True)
    model = ai_methods.create_model(current_arch, input_size, output_size, hidden_layers, 0.4)
    print("MODEL: {} IS CREATED.".format(current_arch))
    print("Input Size: {}, Output Size: {}, Hidden Layers: {}".format(input_size, output_size, hidden_layers))
    print()
    
    print("Training is started in {} mode...".format(my_device))
    optimizer = ai_methods.train_model(model, dataloaders["train"], dataloaders["valid"], train_args.learning_rate, train_args.epochs, 40, my_device)
    print()
    
    print("Training is completed, testing the network...")
    loss, accuracy = ai_methods.validation(model, dataloaders["test"], my_device)
    print("TEST LOSS: {:.3f}...".format(loss))
    print("TEST ACCURACY: {:.2f}%".format(100*accuracy))
    print()
    
    print("Saving model as {}...".format(train_args.save_dir))
    ai_methods.save_model(model, train_args.save_dir, optimizer, current_arch, train_args.epochs, class_to_idx)
    print("Saving is completed!")
    
    return

# Call to main function to run the program
if __name__ == "__main__":
    main()