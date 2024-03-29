# %%-----------------------------------------------------------------------
# Loading required packages
import os, csv  # For handling directories
import matplotlib

matplotlib.use('Agg')
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np  # For storing data as numpy arrays
import pandas as pd
import timeit
from pycm import *
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
# %%-----------------------------------------------------------------------

# importing torch packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image  # For handling the images
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset



# %matplotlib inline


# %%-----------------------------------------------------------------------
# os.system("wget https://s3.amazonaws.com/dataml2/leapGestRecog.tar.gz")
# os.system("mkdir data")
# os.system(tar -xf  leapGestRecog.tar.gz -C ~/MachineLearning2_Project/data
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
    torch.save(model.state_dict(), '../model/model_trained.pth')

    return loss_list

    # plt.plot(loss_list)
    # plt.show()


# %%-----------------------------------------------------------------------

def validate(model, device, validate_loader, criterion, epoch):
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

def test(model, device, test_loader, criterion, epoch):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    loss_list = []
    all_predicted = []
    all_labels = []
    classes = (
    '10_down', '04_fist_moved', '01_palm', '05_thumb', '02_l', '09_c', '08_palm_moved', '07_ok', '03_fist', '06_index')
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            for item in labels.cpu().numpy():
                all_labels.append(item)
            labels = labels.long()
            labels = labels.view(-1, len(labels))[0]
            outputs = model(images)
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            if i % 10 == 0:
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(images), len(test_loader.dataset),
                           100. * i / len(test_loader), loss.item()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for item in predicted.cpu().numpy():
                all_predicted.append(item)

        print('Test Accuracy of the model on the 3000 test images: {} %'.format(100 * correct / total))
        accuracy = 100 * correct / total

        all_labels_array = np.array(all_labels).reshape(-1, )
        all_predicted_array = np.array(all_predicted).reshape(-1, )

        my_dict = dict(list(enumerate(classes)))

        #print(all_labels_array)
        #print(all_predicted_array)
        #print("My dict=", my_dict)
        all_labels_vect = np.vectorize(my_dict.get)(all_labels_array)
        #print(all_labels_vect)
        all_predicted_vect = np.vectorize(my_dict.get)(all_predicted_array)

        # Create CM From Data
        cm1 = ConfusionMatrix(predict_vector=all_predicted_vect, actual_vector=all_labels_vect)

        # Create CM From Data
        #cm1 = ConfusionMatrix(actual_vector=all_labels_array, predict_vector=all_predicted_array)
        cm = confusion_matrix(y_target=all_labels_array, y_predicted=all_predicted_array, binary=False)
        # print(cm.F1)
        # print(cm1)

        # print(type(cm.F1))
        # print(type(cm1))
    return accuracy, loss_list, cm, cm1


# %%-----------------------------------------------------------------------

def main():
    # specify device and choose gpu if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir('../data/leapGestRecog/00/'):
        if not j.startswith('.'):  # If running this code locally, this is to
            # ensure you aren't reading in hidden folders
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
    print(lookup)

    classes = (lookup.keys())

    x_data = []
    label_data = []
    imagecount = 0  # total Image count
    for i in range(0, 10):  # Loop over the ten top-level folders
        for j in os.listdir('../data/leapGestRecog/0' + str(i) + '/'):
            if not j.startswith('.'):  # Again avoid hidden folders
                count = 0  # To tally images of a given gesture
                # loop over the images
                # read in and convert to greyscale
                for k in os.listdir('../data/leapGestRecog/0' + str(i) + '/' + j + '/'):
                    img = Image.open('../data/leapGestRecog/0' +
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

    batch_size_list = [128,64,32]

    results = {}
    resultsDF = []
    f1DF = []

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
        learning_rate_list = [0.1, 0.01, 0.001]
        optimizer_functions_list = ['Adam', 'Adadelta']

        for LEARNING_RATE in learning_rate_list:

            for OPTIMIZER in optimizer_functions_list:
                # create instance of model
                model = CNN().to(device)
                criterion = nn.CrossEntropyLoss()

                if OPTIMIZER == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
                elif OPTIMIZER == 'ASGD':
                    optimizer = torch.optim.ASGD(model.parameters(), lr=LEARNING_RATE)
                elif OPTIMIZER == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                elif OPTIMIZER == 'Adagrad':
                    optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
                elif OPTIMIZER == 'Adadelta':
                    optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
                elif OPTIMIZER == 'RMSProp':
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

                number_epochs_list = [2, 5, 10]

                for NUM_EPOCHS in number_epochs_list:
                    training_loss = []
                    validation_loss = []
                    mean_training_loss = []
                    mean_validation_loss = []

                    for epoch in range(1, NUM_EPOCHS + 1):
                        start = timeit.default_timer()
                        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
                        stop = timeit.default_timer()
                        val_loss = validate(model, device, validate_loader, criterion, epoch)
                        training_loss = training_loss + train_loss
                        validation_loss = validation_loss + val_loss
                        mean_training_loss = mean_training_loss + [np.mean(train_loss)]
                        mean_validation_loss = mean_validation_loss + [np.mean(val_loss)]
                        accuracy, testing_loss, cm, cm1 = test(model, device, test_loader, criterion, epoch)

                    fig1 = plt.figure(figsize=(12, 8))
                    plt.plot(training_loss)
                    plt.plot(validation_loss)
                    plt.xlabel("batch samples", fontsize=20)
                    plt.ylabel("loss", fontsize=20)
                    plt.legend(['training loss', 'validation loss'], loc='upper right', fontsize=20)
                    plt.title("batch-wise training and validation loss for batch size=" + str(BATCH_SIZE)
                              + ", lr=" + str(LEARNING_RATE) +
                              ", optimizer=" + str(OPTIMIZER) + ", num_epochs=" + str(NUM_EPOCHS), fontsize=15)
                    # plt.show()
                    fig1.savefig("batchwise_training_validation_loss_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) +
                                 "_" + str(OPTIMIZER) + "_" + str(NUM_EPOCHS) + ".png")

                    fig2 = plt.figure(figsize=(12, 8))
                    plt.plot(mean_training_loss)
                    plt.plot(mean_validation_loss)
                    plt.xlabel("epochs", fontsize=20)
                    plt.ylabel("mean loss", fontsize=20)
                    plt.legend(['mean training loss', 'mean validation loss'], loc='upper right', fontsize=20)
                    # plt.show()
                    plt.title("epoch-wise mean training and validation loss for batch size=" + str(BATCH_SIZE) +
                              ", lr=" + str(LEARNING_RATE) +
                              ", optimizer=" + str(OPTIMIZER) + ", num_epochs=" + str(NUM_EPOCHS), fontsize=15)
                    fig2.savefig(
                        "epochwise_mean_training_validation_loss_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) +
                        "_" + str(OPTIMIZER) + "_" + str(NUM_EPOCHS) + ".png")

                    fig3 = plt.figure(figsize=(12, 8))
                    plt.plot(training_loss)
                    plt.xlabel("batch samples", fontsize=20)
                    plt.ylabel("loss", fontsize=20)
                    plt.legend(['testing loss'], loc='upper right', fontsize=20)
                    plt.title("testing loss for batch size=" + str(BATCH_SIZE)
                              + ", lr=" + str(LEARNING_RATE) +
                              ", optimizer=" + str(OPTIMIZER) + ", num_epochs=" + str(NUM_EPOCHS), fontsize=15)
                    # plt.show()
                    fig3.savefig("testing_loss_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) +
                                 "_" + str(OPTIMIZER) + "_" + str(NUM_EPOCHS) + ".png")

                    fig4, ax = plot_confusion_matrix(conf_mat=cm)
                    #plt.show()

                    fig4.savefig("confusion_matrix_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) +
                                 "_" + str(OPTIMIZER) + "_" + str(NUM_EPOCHS) + ".png")

                    results[(BATCH_SIZE, LEARNING_RATE, OPTIMIZER, NUM_EPOCHS)] = (
                    round(stop - start, 2), round(accuracy, 2))


                    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
                    for fig in range(1, plt.gcf().number + 1):  ## will open an empty extra figure
                        pdf.savefig(fig)
                    pdf.close()


                df = pd.DataFrame(list(results.items()))
                df1 = pd.DataFrame(df.iloc[:, 0].tolist(),
                                   columns=['batch_size', 'learning_rate', 'optimizer_method', 'num_epochs'])
                df2 = pd.DataFrame(df.iloc[:, 1].tolist(), columns=['time', 'accuracy'])
                df3 = pd.concat([df1, df2], axis=1)

                df3.to_csv("df3.csv")



                df_f1 = pd.DataFrame(list(cm1.F1.items()),columns=['labels', 'f1_score'])

                resultsDF.append(df3)
                f1DF.append(df_f1)

    df_results = pd.concat(resultsDF)

    df_results = df_results.drop_duplicates(keep='first', inplace=False)
    df_results11 = df_results.drop_duplicates(keep='last', inplace=False)

    df_results.to_csv("results.csv")
    df_results11.to_csv("results_last.csv")
    df_results1 = pd.concat(f1DF)
    df_results1.to_csv("f1_score_results.csv")

# %%-----------------------------------------------------------------------

if __name__ == '__main__':
    main()
