# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Setup
- Start by donwloading [pytorch](https://pytorch.org) and [Jupyter Notebook](https://jupyter.org/install)
- To train a CNN on a new dataset run `python train.py data_directory` from the root of this project. 
  - This will create a .pth checkpoint file.
  - The data for training should be ordered: 
    - data_directory/train/classs/Images,  
    - data_directory/valid/classs/Images,  
    - data_directory/test/classs/Images
- To run inference on a trained model use `python predict.py /path/to/image checkpoint.pth`
