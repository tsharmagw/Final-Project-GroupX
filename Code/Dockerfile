From ubuntu:16.04
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN echo which pip3
RUN pip3 install pandas numpy boto3 keras matplotlib opencv-python tensorflow slidingwindow Pillow h5py scikit-image tqdm tensorpack requests dill fire argparse
RUN pip3 install torch torchvision
RUN apt-get -y install python3-tk
RUN apt-get -y install swig
RUN apt-get -y install gcc
RUN apt-get -y install libglib2.0-0
RUN apt-get -y install libsm6
ADD sqs_message_poll.py /home/
ADD model /home/model
ADD script.sh /home/
RUN chmod 775 /home/script.sh
RUN chmod 775 /home/sqs_message_poll.py
CMD ["python3", "/home/sqs_message_poll.py"]


