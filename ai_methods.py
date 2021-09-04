#
# AI METHODS
# Created by: AHMET CELIK
#

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

def get_loader_and_classidx(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = dict()
    data_transforms["train"] = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    data_transforms["valid"] = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    data_transforms["test"] = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    image_datasets = dict()
    image_datasets["train"] = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
    image_datasets["valid"] = datasets.ImageFolder(valid_dir, transform=data_transforms["valid"])
    image_datasets["test"] = datasets.ImageFolder(test_dir, transform=data_transforms["test"])

    dataloaders = dict()
    dataloaders["train"] = torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True)
    dataloaders["valid"] = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32)
    dataloaders["test"] = torch.utils.data.DataLoader(image_datasets["test"], batch_size=32)
    
    class_to_idx = image_datasets["train"].class_to_idx
    return dataloaders, class_to_idx

def create_model(arch, input_size, output_size, hidden_layers, p_drop=0.4):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    my_classifier = nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                  nn.ReLU(),
                                  nn.Dropout(p=p_drop),
                                  nn.Linear(hidden_layers[0], hidden_layers[1]),
                                  nn.ReLU(),
                                  nn.Dropout(p=p_drop),
                                  nn.Linear(hidden_layers[1], output_size),
                                  nn.LogSoftmax(dim=1))
    model.classifier = my_classifier
    return model

def validation(model, loader, device="cpu", criterion=None):
    if criterion is None:
        criterion = nn.NLLLoss()
    loss = 0
    acc = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            loss += criterion(output, labels).item()
            probs = torch.exp(output)
            equality = (labels.data == probs.max(dim=1)[1])
            acc += equality.type(torch.FloatTensor).mean()
    result_loss = loss/len(loader)
    result_acc = (acc.item())/len(loader)
    return result_loss, result_acc

def train_model(model, trainloader, validloader, learning_rate, epochs, print_every=40, device="cpu", optimizer=None):
    # This function trains the model and returns the optimizer
    criterion = nn.NLLLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
    model.to(device)
    for e in range(epochs):
        steps = 0
        running_loss = 0
        model.train()
        
        for images, labels in trainloader:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss, valid_accuracy = validation(model, validloader, device, criterion)
                
                str_print = "Epoch: "+str(e+1)+" of "+str(epochs)+"... Train Loss: "+str(running_loss/print_every)
                str_print += "... Valid Loss: "+str(valid_loss)
                str_print += "... Valid Accuracy: "+str(valid_accuracy)
                print(str_print)
                
                running_loss = 0
                model.train()
    return optimizer

def save_model(model, check_path, optimizer, arch, epochs, class_to_idx):
    model.class_to_idx = class_to_idx
    checkpoint = {"classifier":model.classifier,
                  "state_dict":model.state_dict(),
                  "optimizer": optimizer,
                  "optimizer_state_dict":optimizer.state_dict(),
                  "arch": arch,
                  "class_to_idx":model.class_to_idx,
                  "epochs":epochs}
    torch.save(checkpoint, check_path)
    return

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint["arch"])(pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])
    return model

def process_image(image):
    img = Image.open(image)
    # Resizing:
    w, h = img.size
    resize_ratio = min(w/256, h/256)
    img_resized = img.resize( (int(w//resize_ratio), int(h//resize_ratio)) )
    # Cropping:
    w, h = img_resized.size
    left = (w - 224)//2
    right = left+224
    upper = (h - 224)//2
    lower = upper+224
    img_cropped = img_resized.crop((left, upper, right, lower))
    # PIL to NUMPY
    img_np = np.array(img_cropped)
    img_np = (img_np)/255
    # Normalization & Transpose
    img_norm = (img_np - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    img_trns = img_norm.transpose((2,0,1))
    return torch.FloatTensor([img_trns])

def predict_method(image_path, model, topk, device="cpu"):
    img = process_image(image_path)
    img = img.to(device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model.forward(img)
        ps = torch.exp(output)
        ps_k, ind_k = ps.topk(topk)
        ps_k, ind_k = ps_k.to('cpu').numpy().flatten(), ind_k.to('cpu').numpy().flatten()
    
    idx_to_classes = {v:k for k,v in model.class_to_idx.items()}
    classes = []
    for ind in ind_k:
        classes.append(idx_to_classes[ind])
        
    return ps_k,classes

