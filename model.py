import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json
from PIL import Image

class Model():
    def create_model(self, arch, hidden_units, output_size, dropout = 0.2):
        '''
        Creates a model using a pretrained CNN, and a new classifier

        Args:
            arch (string): The name of the pretrained CNN: densened121, vgg13
            hidden_units (int): The number of hidden units in each of the three hidden layers for the classifer
            output_size (int): The number of output classes to predict
            dropout (float): the dropout percentage
        '''
        # gets the pretrained model specified
        self.model, input_units = self.get_preetrained_model(arch)
        self.arch = arch
        
        # stores data about our model, for saving and loading
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.dropout = dropout
        
        # Freezing the CNN parameters so we wont backprop through them
        for param in self.model.parameters():
            param.requires_gradu = False
            
        # gets a clean classifier
        self.model.classifier = self.create_classifier(input_units, hidden_units, output_size, dropout)
        
        # sets our epochs to 0
        self.epcohs = 0
        
        # defines and sends our model to the cpu by default can be chaned using 'use_gpu' method
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        
    def get_preetrained_model(self, arch):
        '''
        Gets a preetrained model for transfor learning

        Args:
            arch (string): The name of the pretrained CNN: densened121, vgg13

        Returns:
            (model): a preetrained model
            (input_unites): the number of input units the classifier should have
        '''
        if arch == 'densenet121':
            model = models.densenet121(pretrained=True)
            input_units = (list(model.children())[-1]).in_features
            return model, input_units
        elif arch == 'vgg13':
            model = models.vgg13(pretrained=True)
            input_units = model.classifier[0].in_features
            return model, input_units
        else:
            raise Exception('Model not supported, supported models: densened121 or, vgg13')
        
    def create_classifier(self, input_units, hidden_units, output_size, dropout):
        '''
        Creates a custom classifer

        Args:
            input_units (int): The number of input units
            hidden_units (int): the number of hidden units in each hidden layer
            output_size (int): the number of classes to predict
            dropout (float): the dropout procentage

        Returns:
            (classifer): returns a new classifer
        '''
        return nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(input_units, hidden_units)),
            ('relu_1', nn.ReLU()),
            ('dropout_1', nn.Dropout(dropout)),
            ('fc_2', nn.Linear(hidden_units, hidden_units)),
            ('relu_2', nn.ReLU()),
            ('dropout_2', nn.Dropout(dropout)),
            ('fc_3', nn.Linear(hidden_units, output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
    def train(self, epochs, data_dataloaders, learning_rate = 0.003, print_every_step = 20):
        '''
        Trains the model
        
        Args:
            epochs (int): The number of epochs to train on
            data_dataloaders (dataloader): the dataloader array
            learning_rate (float): the learning rate of the optimizer
            print_every_step (int): steps between loss will be printed
        '''
        # defines our criterion and optimizer if their is none
        if ('criterion' in vars()) == False:
            self.criterion = nn.NLLLoss()
        if ('optimizer' in vars()) == False:
            self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        
        # sets the new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # counting number of steps
        steps = 0 
        
        # sets our model to train mode
        self.model.train()
        # sends our model to the device (cpu/gpu)
        self.model = self.model.to(self.device)
        # runs through set amount of epochs
        for epoch in range(epochs):
            # holds our runnning loss
            running_loss = 0
            # counts number of epochs
            self.epochs = 1

            # loops over our train data
            for inputs, labels in data_dataloaders['train']:
                # add 1 to the number of steps done
                steps += 1

                # moves input- and labels- tensors to the device (cpu/gpu)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # resets our optimizer
                self.optimizer.zero_grad()
                # gets our log probabilities from our model
                logps = self.model.forward(inputs)
                # calculates the loss using our criterion
                loss = self.criterion(logps, labels)
                # backprobagates to update our weights
                loss.backward()
                self.optimizer.step()

                # adds our loss to the total loss, between validation
                running_loss += loss.item()

                # calculates our models current accuracy each x steps
                if steps % print_every_step == 0:
                    # holds our test loss
                    test_loss = 0
                    # holds our total accuracy
                    accuracy = 0

                    # sets our model to eval, to stop dropout
                    self.model.eval()

                    # stops our optimizer for better performance
                    with torch.no_grad():
                        # loops through our validation data
                        for inputs, labels in data_dataloaders['valid']:
                            # moves input- and labels- tensors to the device (cpu/gpu)
                            inputs, labels = inputs.to(self.device), labels.to(self.device)

                            # gets our log probabilities from our model
                            logps = self.model.forward(inputs)
                            # calculates the loss using our criterion
                            batch_loss = self.criterion(logps, labels)
                            # adds our loss to the total loss validation loss
                            test_loss += batch_loss.item()

                            # ---Calcualtes the accuracy
                            # converts to probabilities
                            ps = torch.exp(logps)
                            # gets the top prediction
                            top_p, top_class = ps.topk(1, dim=1)
                            # Check if it is equal to the label
                            equals = top_class == labels.view(*top_class.shape)
                            # adds our accuracy this step to the total validation accuracy
                            accuracy += torch.mean(equals.type(torch.FloatTensor))

                    # Prints out data about our current model accuracy and losses
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Valid loss: {running_loss/print_every_step:.3f}.. "
                          f"Valid loss: {test_loss/len(data_dataloaders['valid']):.3f}.. "
                          f"Valid accuracy: {accuracy/len(data_dataloaders['valid']):.10f}")

                    # resets our running loss
                    running_loss = 0
                    # resets our model to train mode
                    self.model.train()
                    
    def validate(self, dataloaders):
        '''
        Prints out the accuracy of the model

        Args:
            dataloaders (dataloader): the dataloader array
        '''
        
        # holds our total test loss
        test_loss = 0
        # holds our total accuracy
        accuracy = 0
        # sends our model to the device (cpu/gpu)
        self.model = self.model.to(self.device)
        # sets our model to eval mode
        self.model.eval()

        # stops our optimizer for better performance
        with torch.no_grad():
            # loops over our test data
            for inputs, labels in dataloaders['test']:
                # moves input- and labels- tensors to the device (cpu/gpu)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # gets our log probabilities from our model
                logps = self.model.forward(inputs)

                # --- Calcualtes the accuracy
                # converts to probabilities
                ps = torch.exp(logps)
                # gets the top prediction
                top_p, top_class = ps.topk(1, dim = 1)
                # Check if it is equal to the label
                equals = top_class == labels.view(*top_class.shape)
                # adds our accuracy this step to the total test accuracy
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        # prints our final result
        print('Test accuracy: {}'.format(accuracy / len(dataloaders['test'])))
        
    def save_model(self, class_to_idx, path):
        '''
        Saves the model

        Args:
            class_to_idx (): the class_to_idx from the dataset
            path (string): the path to save the model
        '''
        torch.save({
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'optimizer_lr' : self.optimizer.param_groups[0]['lr'],
            'class_to_idx': class_to_idx,
            'epoch' : self.epochs,
            'arch': self.arch,
            'hidden_units' : self.hidden_units,
            'dropout': self.dropout,
            'output_size': self.output_size
        }, path)
           
    def load_model(self, path):
        '''
        Loads a model from a path

        Args:
            path (string): the path to the saved model
        '''
        # loads the checkpoint
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

        # creates a new model
        self.create_model(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['output_size'], checkpoint['dropout'])
        
        # loads our weights and a class_to_idx.
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']

        # recreating our optimizer
        self.optimizer = optim.Adam(self.model.classifier.parameters(), checkpoint['optimizer_lr'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # recording how many epochs our network has completed training on
        self.epochs = checkpoint['epoch']
        
    def use_gpu(self, value):
        '''
        Toggles the model to run on the gpu

        Args:
            value (boolean): Should run on gpu
        '''
        # defines the device which our classifier should run on
        if value == True:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.model = self.model.to(self.device)
            else:
                raise Exception('Cuda acceleration not supported on hardware')
        else:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
       
    def process_image(self, image):
        '''
        Scales, crops, and normalizes a PIL image for a PyTorch model,

        Args:
            value (boolean): Should run on gpu
        Returns:
            (Numpy array): returns a numpy array
        '''
        # creates the pre proccessing transforms
        resize = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # applying transforms and returns the image
        return resize(image)
    
    def predict(self, image_path, category_names_dir=None,topk=5):
        '''
        Predict the class (or classes) of an image using a trained deep learning model.

        Args:
            image_path (string): path the the image
            category_names_dir (string): Should run on gpu
            topk (int): Should run on gpu
            
        Returns:
            (top_p): the top k probability
            (top_classes): the top classes
        '''
        # opens the image and applies preproccessing
        image = self.process_image(Image.open(image_path))
        # unsqueezes it
        image = image.unsqueeze_(0)
        # sends it to our device (cpu/gpu)
        image = image.to(self.device)
        self.model = self.model.to(self.device)

        # sets our model to eval mode, to disable dropout
        self.model.eval()
        # stops our optimizer for better performance
        with torch.no_grad():
            # gets our log probabilities from our model
            logps = self.model.forward(image)

        # converts to probabilities
        ps = torch.exp(logps)

        # Get the top 5 probabilities and classes
        prop, classes = ps.topk(topk, dim=1)

        # Get the first items in the tensor list which contains the probs and classes
        top_p = prop.tolist()[0]
        top_classes = classes.tolist()[0]

        if (category_names_dir != None):
            with open(category_names_dir, 'r') as f:
                # sets up a list to hold our labels
                labels = []
                # reverses our class_to_idx which our model holds
                idx_to_class = {v : k for k, v in self.model.class_to_idx.items()}
                
                # gets our json file
                cat_to_name = json.load(f)
                
                # loops through each prediction
                for c in top_classes:
                    # adds the name which our model did predict
                    labels.append(cat_to_name[idx_to_class[c]])

                # returns our top k probabilities and labels as a lists.
                return top_p, labels
        else:
            # returns our top k probabilities and classes as a lists.
            return top_p, top_classes