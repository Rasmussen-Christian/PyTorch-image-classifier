import torch
from torchvision import datasets, transforms

class Data_manager:
    
    def get_dataloaders(self, directory):        
        '''
        Returns 3 dataLoaders: train, valid, and test
            folder structure:
                Directory/test/class_name/images
                Directory/valid/class_name/images
                Directory/test/class_name/images

        Args:
            directory (string): the path to the data

        Returns:
            (DataLoader): an array of dataloaders
        '''

        # Defines the transforms for the training, validation, and testing sets
        self.data_transforms = {
            'train': transforms.Compose([
                        transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ]),
            'valid': transforms.Compose([
                        transforms.Resize(255),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ]),
            'test': transforms.Compose([
                        transforms.Resize(255),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
        }


        # Loads the datasets with ImageFolder
        self.data_datasets = {
            'train': datasets.ImageFolder(directory + '/train', transform=self.data_transforms['train']),
            'valid': datasets.ImageFolder(directory + '/valid', transform=self.data_transforms['valid']),
            'test': datasets.ImageFolder(directory + '/test', transform=self.data_transforms['test'])
        }

        # TODO: Using the image datasets and the trainforms, define the dataloaders
        self.data_dataloaders = {
            'train': torch.utils.data.DataLoader(self.data_datasets['train'], batch_size=64, shuffle=True),
            'valid': torch.utils.data.DataLoader(self.data_datasets['valid'], batch_size=64),
            'test': torch.utils.data.DataLoader(self.data_datasets['test'], batch_size=64)
        }
        
        return self.data_dataloaders