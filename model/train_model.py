#%%-----------------------------------------------------------------------
# Loading required packages
import os  # For handling directories
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np  # For storing data as numpy arrays

#%%-----------------------------------------------------------------------

# importing torch packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image  # For handling the images
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# %matplotlib inline

# %%-----------------------------------------------------------------------
# specify the class for preprocessing the data
class DatasetProcessing(Dataset):
    """
    This function is used to initialise the class variables - transform, data, target

    """

    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.data = data.reshape((-1, 120, 320)).astype(np.uint8)[:, :, :, None]
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.target[index]

    def __len__(self):
        return len(list(self.data))


# %%-----------------------------------------------------------------------

# specify the model class
class CNN(nn.Module):
    '''
    Here we are using CNN model with three conv layers with maxpool
    and relu as transfer/activation function
    for fully connected layer again we have used relu activation function
    '''

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# %%-----------------------------------------------------------------------
# Train the Model

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    loss_list = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.long()
        labels = labels.view(-1, len(labels))[0]
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))
    # Save the model checkpoint
    torch.save(model.state_dict(), './model/model_trained.pth')

    return loss_list

    # plt.plot(loss_list)
    # plt.show()


# %%-----------------------------------------------------------------------

def validate(model, device, validate_loader, optimizer, criterion, epoch):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(validate_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.long()
            labels = labels.view(-1, len(labels))[0]
            outputs = model(images)
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            if i % 10 == 0:
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(images), len(validate_loader.dataset),
                           100. * i / len(validate_loader), loss.item()))

        return loss_list


# %%-----------------------------------------------------------------------

def test(model, device, test_loader):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.long()
            labels = labels.view(-1, len(labels))[0]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 3000 test images: {} %'.format(100 * correct / total))


# %%-----------------------------------------------------------------------

def main():
    # specify device and choose gpu if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir('./data/leapGestRecog/00/'):
        if not j.startswith('.'):  # If running this code locally, this is to
            # ensure you aren't reading in hidden folders
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
    print(lookup)

    x_data = []
    label_data = []
    imagecount = 0  # total Image count
    for i in range(0, 10):  # Loop over the ten top-level folders
        for j in os.listdir('./data/leapGestRecog/0' + str(i) + '/'):
            if not j.startswith('.'):  # Again avoid hidden folders
                count = 0  # To tally images of a given gesture
                # loop over the images
                # read in and convert to greyscale
                for k in os.listdir('./data/leapGestRecog/0' + str(i) + '/' + j + '/'):
                    img = Image.open('./data/leapGestRecog/0' +
                                     str(i) + '/' + j + '/' + k).convert('L')
                    img = img.resize((320, 120))
                    arr = np.array(img)
                    x_data.append(arr)
                    count = count + 1

                y_values = np.full((count, 1), lookup[j])
                label_data.append(y_values)
                imagecount = imagecount + count
    x_data = np.array(x_data, dtype='float32')
    label_data = np.array(label_data)
    label_data = label_data.reshape(imagecount, 1)  # Reshape to be the correct size

    # check the shape of train data
    print(x_data.shape)
    print(label_data)

    # divide the data into train, validation and test
    x_train, x_valid_test, y_train, y_valid_test = train_test_split(x_data, label_data, test_size=0.3)
    x_validate, x_test, y_validate, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.5)

    # check the shape of train data
    print(x_train.shape)
    print(y_train.shape)

    # check the shape of validation data
    print(x_validate.shape)
    print(y_validate.shape)

    # check the shape of test data
    print(x_test.shape)
    print(y_test.shape)

    batch_size_list = [64]

    for BATCH_SIZE in batch_size_list:

        # specify the transformation
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        # perform the pre-processing on train data
        data_train = DatasetProcessing(x_train, y_train, transform)
        # load the train data
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        # specify the transformation
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        data_validate = DatasetProcessing(x_validate, y_validate, transform)
        validate_loader = torch.utils.data.DataLoader(data_validate, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        # specify the transformation
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        data_test = DatasetProcessing(x_test, y_test, transform)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        # specify the number of epochs and learning rate
        learning_rate_list = [0.01]
        optimizer_functions_list = ['Adam']

        for LEARNING_RATE in learning_rate_list:

            for OPTIMIZER in optimizer_functions_list:
                # create instance of model
                model = CNN().to(device)
                criterion = nn.CrossEntropyLoss()

                if (OPTIMIZER == 'SGD'):
                    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
                elif (OPTIMIZER == 'ASGD'):
                    optimizer = torch.optim.ASGD(model.parameters(), lr=LEARNING_RATE)
                elif (OPTIMIZER == 'Adam'):
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                elif (OPTIMIZER == 'Adagrad'):
                    optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
                elif (OPTIMIZER == 'Adadelta'):
                    optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
                elif (OPTIMIZER == 'RMSProp'):
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
                # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                number_epochs_list = [2]

                for NUM_EPOCHS in number_epochs_list:
                    training_loss = []
                    validation_loss = []
                    mean_training_loss = []
                    mean_validation_loss = []

                    for epoch in range(1, NUM_EPOCHS + 1):
                        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
                        val_loss = validate(model, device, validate_loader, optimizer, criterion, epoch)
                        training_loss = training_loss + train_loss
                        validation_loss = validation_loss + val_loss
                        mean_training_loss = mean_training_loss + [np.mean(train_loss)]
                        mean_validation_loss = mean_validation_loss + [np.mean(val_loss)]
                        test(model, device, test_loader)

                    fig1 = plt.figure()
                    plt.plot(training_loss)
                    plt.plot(validation_loss)

                    fig1.savefig("batchwise_training_validation_loss_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) +
                                 "_" + str(OPTIMIZER) + "_" + str(NUM_EPOCHS) + ".png")

                    fig2 = plt.figure()
                    plt.plot(mean_training_loss)
                    plt.plot(mean_validation_loss)

                    fig2.savefig(
                        "epochwise_mean_training_validation_loss_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) +
                        "_" + str(OPTIMIZER) + "_" + str(NUM_EPOCHS) + ".png")

                    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
                    for fig in range(1, plt.gcf().number + 1):  ## will open an empty extra figure
                        pdf.savefig(fig)
                    pdf.close()


# %%-----------------------------------------------------------------------

if __name__ == '__main__':
    main()
