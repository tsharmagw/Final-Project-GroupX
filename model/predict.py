import io

from PIL import Image, ImageOps, ImageDraw, ImageFont
import requests
import boto3
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")


input_size = 1 * 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 50
batch_size = 10
learning_rate = 0.01



# Random cat img taken from Google
#IMG_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg'
s3_client = boto3.resource('s3',
	    	aws_access_key_id='AKIAIAQS3KGWKQL4JJ7A',
	    	aws_secret_access_key='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',
	    	region_name="us-east-1")

key,bucket=open("/tmp/message_data.txt", "r").readline().split(";")
#key='sn.jpg'
#bucket='flaskdataml2'
download_path = '/tmp/{}'.format(key)
s3_client.Bucket(bucket).download_file(key, download_path)
#logger.info("Data downloaded from s3 FIle name is "+str(key) )
IMG_URL_LOCAL = download_path
# Class labels used when training VGG as json, courtesy of the 'Example code' link above.
#LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

# Let's get our class labels.
#response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
#labels = {int(key): value for key, value in response.json().items()}

# Let's get the cat img.
#response = requests.get(IMG_URL)
#img = Image.open(io.BytesIO(response.content))  # Read bytes and store as an img.

img=Image.open(IMG_URL_LOCAL)

# Let's take a look at this cat!

#img.show()

img = torchvision.transforms.functional.to_grayscale(img,1)

#img.show()

#img =torch.sub(torch.FloatTensor(255)-Variable(img))
#img.show()




min_img_size = 28  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


img = transform_pipeline(img)

img=img.resize_((1,28,28))

print(img.size())

# PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels, height, width).
# Currently however, we have (num color channels, height, width); let's fix this by inserting a new axis.
#img = img.unsqueeze(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims.

# Now that we have preprocessed our img, we need to convert it into a
# Variable; PyTorch models expect inputs to be Variables. A PyTorch Variable is a
# wrapper around a PyTorch Tensor.



class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = Net(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('/home/model/model_Q5_cpuSGD.pth'))
#model.load_state_dict(torch.load('./model/model_Q5_cpuSGD.pth'))

print(model)




images = Variable(img.view(-1,28 * 28))

print(images.size())

outputs = model(images)
print(outputs.size())
_, predicted = torch.max(outputs.data, 1)

print(predicted.size())

print(predicted)

#predicted = predicted.data.cpu().numpy().argmax()



classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
pred_class=classes[predicted]
#adding label to image
img=Image.open(IMG_URL_LOCAL)
bimg = ImageOps.expand(img, border=20)
draw = ImageDraw.Draw(bimg)
draw.text((28, 20),pred_class,(255,255,255))
bimg.save('/tmp/sample.jpg')
bucket1='flaskdataml2output'
s3_client.Bucket(bucket1).upload_file('/tmp/sample.jpg',Key='pred_class'+".jpg")

