import tensorflow as tf
import os
import numpy as np
import cv2

root = '/home/rishhanth/Documents/gen_codes/CSRNet-tf/train.tfrecords'
batch_size = 1

def parse_records(recordfile):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(recordfile, feature)
    image = tf.reshape(tf.decode_raw(features['train/image'], tf.float32),[224,224,3]) 
    label = tf.reshape(tf.decode_raw(features['train/label'], tf.float32),[224,224,1]) 
   
    return image,label

def input_data():
    train_dataset = tf.data.TFRecordDataset(root)
    train_dataset = train_dataset.map(parse_records,num_parallel_calls=4)
    train_dataset = train_dataset.shuffle(100).repeat()
    train_dataset = train_dataset.batch(8)
    return train_dataset.make_one_shot_iterator()

if __name__ == '__main__':
    iterator = input_data()
    images = iterator.get_next()
    with tf.Session() as sess:
        while True:
            listi = sess.run(images)
            print(listi[0].shape)
            print(listi[1].shape)