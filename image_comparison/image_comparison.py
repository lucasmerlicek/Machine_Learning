import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import models
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract
    the embeddings.
    """
    # Define a transform to pre-process the images
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(root="dataset", transform=train_transforms)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, num_workers=8)

    model = nn.Module()
    embeddings = []
    embedding_size = 2048
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # Use the model to extract the embeddings, remove the last layers of the 
    # model to access the embeddings the model generates. 

    pretrained_model = models.resnet50(pretrained=True)
    # remove the last layer
    embedding_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
    embedding_extractor = embedding_extractor.to(device)

    with torch.no_grad():  # no need to compute gradients for embedding extraction
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            embeddings_batch = embedding_extractor(images)
            embeddings_batch = embeddings_batch.view(embeddings_batch.size(0), -1)
            start_index = i * train_loader.batch_size
            end_index = start_index + embeddings_batch.size(0)
            embeddings[start_index:end_index] = embeddings_batch.cpu().numpy()

    np.save('dataset/embeddings_resnet50.npy', embeddings)
def normalize_embeddings(embeddings):
    l2_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_emb = embeddings/l2_norm
    return norm_emb
def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings_resnet50.npy')
    # normalize the embeddings across the dataset
    embeddings = normalize_embeddings(embeddings)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i].replace('food\\', '')] = embeddings[i]

    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    example_key = list(file_to_embedding.keys())[0]

    return X, y

def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader
class Net(nn.Module):
    """
    The model class, which defines the classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc0 = nn.Linear(6144, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 13
    # define a loss function, optimizer and proceed with training.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00008)

    X_train, y_train = train_loader.dataset.tensors
    X_train, X_val, y_train, y_val = train_test_split(X_train.numpy(), y_train.numpy(), test_size=0.1, random_state=33)

    train_loader = create_loader_from_np(X_train, y_train, train=True, batch_size=64)
    val_loader = create_loader_from_np(X_val, y_val, train=True, batch_size=64)

    previous_val_loss = 1.0
    for epoch in range(n_epochs):        
        for [X, y] in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y.type(torch.float))
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_preds = 0.0
        with torch.no_grad():
            for [X_val, y_val] in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs_val = model(X_val).squeeze()
                loss_val = criterion(outputs_val, y_val.type(torch.float))
                outputs_val[outputs_val >= 0.5] = 1
                outputs_val[outputs_val < 0.5] = 0
                correct_preds += (outputs_val == y_val.type(torch.float)).float().sum()
                val_loss += loss_val.item()

        val_loss /= len(val_loader)
        accuracy = correct_preds/len(val_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        # Switch back to train mode
        model.train()
    return model
def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results_resnet50_n13.txt", predictions, fmt='%i')


# Main function.
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings_resnet50.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")