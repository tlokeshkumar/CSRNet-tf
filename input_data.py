import tensorflow as tf
import os
import numpy as np
import random
img_rows = 512
img_cols = 512
fac = 8
def _corrupt_brightness(image, mask):
    """Radnomly applies a random brightness change."""
    cond_brightness = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image, mask


def _corrupt_contrast(image, mask):
    """Randomly applies a random contrast change."""
    cond_contrast = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _corrupt_saturation(image, mask):
    """Randomly applies a random saturation change."""
    cond_saturation = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask

def parse_records(recordfile):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(recordfile, feature)
    image = tf.reshape(tf.decode_raw(features['train/image'], tf.float32),[224,224,3]) 
    image = image - tf.constant([103.99, 116.779, 123.68])
    image = tf.image.resize_images(image, [img_rows, img_cols])
    label = tf.reshape(tf.decode_raw(features['train/label'], tf.float32),[224,224,1]) 
    label = tf.image.resize_images(label, [img_rows//fac, img_cols//fac])
    return image,label

def _flip_left_right(image, mask):
    """Randomly flips image and mask left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)
    mask = tf.image.random_flip_left_right(mask, seed=seed)

    return image, mask

def _crop_random(image, mask):
    """Randomly crops image and mask in accord."""
    seed = random.random()
    cond_crop_image = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    cond_crop_mask = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

    image = tf.cond(cond_crop_image, lambda: tf.random_crop(
        image, [int(img_rows * 0.85), int(img_cols * 0.85), 3], seed=seed), lambda: tf.identity(image))
    mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
        mask, [int(img_rows//fac * 0.85), int(img_cols//fac * 0.85), 1], seed=seed), lambda: tf.identity(mask))
    image = tf.expand_dims(image, axis=0)
    mask = tf.expand_dims(mask, axis=0)

    image = tf.image.resize_images(image, [img_rows, img_cols])
    mask = tf.image.resize_images(mask, [img_rows//fac, img_cols//fac])

    image = tf.squeeze(image, axis=0)
    mask = tf.squeeze(mask, axis=0)

    return image, mask

def input_data(TFRecordfile = '/home/rishhanth/Documents/gen_codes/CSRNet-tf/train.tfrecords',batch_size = 8, augment = True, num_threads=2, prefetch =30):
    train_dataset = tf.data.TFRecordDataset(TFRecordfile)
    train_dataset = train_dataset.map(parse_records,num_parallel_calls=num_threads)
    if augment:
        train_dataset = train_dataset.map(_corrupt_brightness,
                        num_parallel_calls=num_threads).prefetch(prefetch)

        train_dataset = train_dataset.map(_corrupt_contrast,
                        num_parallel_calls=num_threads).prefetch(prefetch)

        train_dataset = train_dataset.map(_corrupt_saturation,
                        num_parallel_calls=num_threads).prefetch(prefetch)

        train_dataset = train_dataset.map(_crop_random,
                        num_parallel_calls=num_threads).prefetch(prefetch)

        train_dataset = train_dataset.map(_flip_left_right,
                        num_parallel_calls=num_threads).prefetch(prefetch)
    train_dataset = train_dataset.shuffle(prefetch).repeat()
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset.make_one_shot_iterator()

if __name__ == '__main__':
    iterator = input_data()
    images = iterator.get_next()
    with tf.Session() as sess:
        while True:
            listi = sess.run(images)
            print(listi[0].shape)
            print(listi[1].shape)
