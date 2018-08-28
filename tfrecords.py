import tensorflow as tf
import cv2
import numpy as np
from random import shuffle
import glob
import sys
import os

root = '/home/rishhanth/Documents/gen_codes/CSRNet-tf/ShanghaiTech/part_A/train_data/images/'

def get_filenames():
    filenames = os.listdir(root)
    image_files = []
    label_files = []
    for i in filenames:
        im_file = os.path.join(root,i)
        image_files.append(im_file)
        label_files.append(im_file.replace('IMG_','LAB_').replace('.jpg','.npy').replace('images','labels'))
    return image_files,label_files

shuffle_data = True  # shuffle the addresses before saving
# read addresses and labels from the 'train' folder
train_addrs,train_labels = get_filenames()
# to shuffle data
if shuffle_data:
    c = list(zip(train_addrs, train_labels))
    shuffle(c)
    train_addrs, train_labels = zip(*c)

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def load_labels(addr):
    lab = np.load(addr)
    lab.astype(np.float32)
    lab = np.array(lab)
    lab = cv2.resize(lab,(224,224), interpolation=cv2.INTER_CUBIC)
    lab.astype(np.float32)    
    return lab

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = '/home/rishhanth/Documents/gen_codes/CSRNet-tf/train.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 10:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = load_labels(train_labels[i])
    # Create a feature
    feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()