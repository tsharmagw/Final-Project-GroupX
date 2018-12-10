#@author- Tejasvi Sharma
'''The purpose of this is to take data from 
'''

#Loading required packages
import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image , ImageOps, ImageDraw, ImageFont# For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting

#importing torch packages
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import boto3

#This class to transform the data, so that it can be loaded to test Loader
class DatasetProcessing(Dataset):
    def __init__(self, data, target, transform=None): #used to initialise the class variables - transform, data, target
        self.transform = transform
        self.data = data.reshape((-1,120,320)).astype(np.uint8)[:,:,:,None]
        self.target = torch.from_numpy(target).float() # needs to be in torch.LongTensor dtype
    def __getitem__(self, index): #used to retrieve the X and y index value and return it
        return self.transform(self.data[index]), self.target[index]
    def __len__(self): #returns the length of the data
        return len(list(self.data))
 
#CNN module, this is network architecture of model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=2),
            nn.BatchNorm2d(32), #undertand, BatchNorm2D, why we have used it
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer3=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=2),
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
        out=self.classifier(out)
        return out


if __name__=="__main__":

    #code to read message from queue needs to be added
    class_dict={9:'palm',7:'L shape',8:'fist',4:'fist_moved',6:'thumb',2:'index_finger',0:'ok',3:'palm_moved',5:'C shape',1:'Down'}
    s3_client = boto3.resource('s3',
            aws_access_key_id='AKIAIAQS3KGWKQL4JJ7A',
            aws_secret_access_key='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',
            region_name="us-east-1")

    key,bucket=open("/tmp/message_data.txt", "r").readline().split(";")

    download_path = '/tmp/{}'.format(key)
    s3_client.Bucket(bucket).download_file(key, download_path)
    IMG_URL_LOCAL = download_path

    img_array = []
    # Read in and convert to greyscale
    img=Image.open(IMG_URL_LOCAL).convert('L')
    img.show()

    img = img.resize((320, 120))
    arr = np.array(img)
    img_array.append(arr)
    img_array1 = np.array(img_array, dtype = 'float32')

    model_test = CNN()
    model_test.load_state_dict(torch.load('/home/model/model_trained_cpu.pth'))


    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    dset_test = DatasetProcessing(img_array1,np.array([[1]]), transform)
    test_loader = torch.utils.data.DataLoader(dset_test, batch_size=1,shuffle=False,num_workers=1) #meaning of num_workers


    for i,(images,label) in enumerate(test_loader):
        images = Variable(images)
        outputs = model_test(images)
        _, predicted = torch.max(outputs.data, 1)
        pred_class=class_dict[predicted.cpu().numpy()[0]]

    #adding label to image
    print(pred_class)
    img=Image.open(IMG_URL_LOCAL)
    bimg = ImageOps.expand(img, border=20)
    draw = ImageDraw.Draw(bimg)
    draw.text((320, 20),pred_class,fill=(255))
    bimg.save('/tmp/sample.jpg')
    bucket1='flaskdataml2output'
    s3_client.Bucket(bucket1).upload_file('/tmp/sample.jpg',Key=key+"-"+pred_class+".jpg")


