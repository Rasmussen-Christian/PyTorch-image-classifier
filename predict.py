import argparse

from model import Model

def main():
    # set up our command line arguments
    parser = argparse.ArgumentParser(description='Set arguments for training of pytorch image recognotion model')
    parser.add_argument('image_dir', type = str, help='path to image to run model on')
    parser.add_argument('checkpoint_dir', type = str, help='path to the model checkpoint')
    parser.add_argument('--top_k', type = int, default=5, help='return top k probabilities')
    parser.add_argument('--category_names', type = str, help='json file containing the conversion from classes to the real labels')
    parser.add_argument('--gpu', type = bool, default=False, help='Should pytorch use a gpu for better performance')
    
    # gets our arguments from the command line
    in_arg = parser.parse_args()
    
    # creates a Model class
    model = Model()
    
    # loads a checkpoint from specified path
    model.load_model(in_arg.checkpoint_dir)
    
    # sends our model to the gpu if requested
    model.use_gpu(in_arg.gpu)
    
    # Gets our top  image prediction from our model
    probs, classes = model.predict(in_arg.image_dir, category_names_dir=in_arg.category_names, topk=in_arg.top_k)

    # prints out our data
    print('model probabilities: {}'.format(probs))
    print('model classes: {}'.format(classes))
    
# Call to main function to run the program
if __name__ == "__main__":
    main()