import argparse
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K_B
from input_data import input_data
from csrnet import create_full_model,loss_funcs
from os.path import exists
import coloredlogs
from colorize import colorize

parser = argparse.ArgumentParser(description="Inputs to the code")

parser.add_argument("--input_record_file",type=str,help="path to TFRecord file with training examples")
parser.add_argument("--batch_size",type=int,default=16,help="Batch Size")
parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
parser.add_argument("--ckpt_savedir",type = str,default='./checkpoints/model_ckpt',help="path to save checkpoints")
parser.add_argument("--load_ckpt",type = str,default='./checkpoints',help="path to load checkpoints from")
parser.add_argument("--save_freq",type = int,default=100,help="save frequency")
parser.add_argument("--display_step",type = int,default=1,help="display frequency")
parser.add_argument("--summary_freq",type = int,default=100,help="summary writer frequency")
parser.add_argument("--no_epochs",type=int,default=10,help="number of epochs for training")

args = parser.parse_args()
no_iter_per_epoch = np.ceil(30000/args.batch_size)
img_rows = 256
img_cols = 256
fac = 8
TFRecord_file = args.input_record_file

if __name__ == '__main__':

    runopts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    coloredlogs.install(level='DEBUG')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    with tf.Graph().as_default():
        init = tf.global_variables_initializer()
    
        iterator = input_data(TFRecord_file,batch_size=args.batch_size)
        images,labels = iterator.get_next()
        labels_resized = tf.image.resize_images(labels,[img_rows//fac, img_cols//fac])

        model_A = create_full_model(images, 'b')
        
        print (model_A.summary())

        tf.summary.image('input-image', images)
        tf.summary.image('label', tf.map_fn(lambda img: colorize(img, cmap='jet'), labels))
        tf.summary.image('predict', tf.map_fn(lambda img: colorize(img, cmap='jet'), model_A.output))
        
        loss_A = loss_funcs(model_A, labels_resized)

        global_step_tensor = tf.train.get_or_create_global_step()

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-6)
        opA = optimizer.minimize(loss_A,global_step=global_step_tensor)
        
    with K_B.get_session() as sess:
        
        sess.run(init)
        summary_writer = tf.summary.FileWriter(args.log_directory, sess.graph)    
        summary = tf.summary.merge_all()
        
        saver = tf.train.Saver()

        tf.logging.info('Tensorboard logs will be written to ' + str(args.log_directory))

        if args.load_ckpt is not None:

            if exists(args.load_ckpt):
                if tf.train.latest_checkpoint(args.load_ckpt) is not None:
                    tf.logging.info('Loading Checkpoint from '+ tf.train.latest_checkpoint(args.load_ckpt))
                    saver.restore(sess, tf.train.latest_checkpoint(args.load_ckpt))

                else:
                    tf.logging.info('Training from Scratch -  No Checkpoint found')
        
        else:
            tf.logging.info('Training from scratch')

        tf.logging.info('Training with Batch Size %d for %d epochs'%(args.batch_size,args.no_epochs))

        while True:    
        # Training Iterations Begin
            global_step,_ = sess.run([global_step_tensor,opA],options = runopts)
            out_a = sess.run(model_A.output)
            if global_step%(args.display_step)==0:
                loss_val = sess.run([loss_A],options = runopts)
                tf.logging.info('Iteration: ' + str(global_step) + ' Loss: ' +str(loss_val))
            
            if global_step%(args.summary_freq)==0:
                tf.logging.info('Summary Written')
                summary_str = sess.run(summary)
                summary_writer.add_summary(summary_str, global_step)
            
            if global_step%(args.save_freq)==0:
                saver.save(sess,args.ckpt_savedir,global_step=tf.train.get_global_step())
            
            if np.floor(global_step/no_iter_per_epoch) == args.no_epochs:
                break
