import argparse

from data_manager import Data_manager
from model import Model

def main():
    # set up our command line arguments
    parser = argparse.ArgumentParser(description='Set arguments for training of pytorch image recognotion model')
    parser.add_argument('data_directory', type = str, help='select directory for images to train on')
    parser.add_argument('--save_dir', type = str, default = 'model_checkpoint.pth', help='select directory for saving of model')
    parser.add_argument('--arch', type = str, default='densenet121', help='select architecture')
    parser.add_argument('--learning_rate', type = float, default='.003', help='select learning rate')
    parser.add_argument('--hidden_units', type = int, default='256', help='select number of hidden units for each of 3 layers')
    parser.add_argument('--epochs', type = int, default=5, help='set number of epochs for training')
    parser.add_argument('--dropout', type = float, default=.2, help='change of dropout')
    parser.add_argument('--print_every', type = int, default=10, help='How often should the model print out training logs')
    parser.add_argument('--output_size', type = int, default=102, help='How many categories should it predict, should be equal to training data')
    parser.add_argument('--gpu', type = bool, default=False, help='Should pytorch use a gpu for better performance')
    
    # gets our arguments from the command line
    in_arg = parser.parse_args()
    
    # instantiates the data_manager class
    data_manager = Data_manager()
    
    # gets our dataloaders from the data_manager class, using our arg input as the directory
    dataloaders = data_manager.get_dataloaders(in_arg.data_directory)
    
    # creates a Model class
    model = Model()
    
    # creates a new model to train on
    model.create_model(in_arg.arch, in_arg.hidden_units, in_arg.output_size, in_arg.dropout)
    
    # sends our model to the gpu if requested
    model.use_gpu(in_arg.gpu)
    
    # trains our model
    model.train(in_arg.epochs, dataloaders, learning_rate = in_arg.learning_rate, print_every_step = in_arg.print_every)
    
    # tests our trained models accuracy
    model.validate(dataloaders)
    
    # saves our trained model for use later
    model.save_model(data_manager.data_datasets['train'].class_to_idx, in_arg.save_dir)
    

# Call to main function to run the program
if __name__ == "__main__":
    main()