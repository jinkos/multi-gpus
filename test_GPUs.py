from __future__ import print_function

import tensorflow as tf
import argparse
import time
import sys

from keras import optimizers,losses
from keras import backend as K

import ratio_training_utils
import gpu_maxing_model

def print_devices():
  devices = K.get_session().list_devices()
  for d in devices:
    splits = d.name.split(':')
    print("{}:{}".format(splits[-2],splits[-1]))

def arg_config(do_print=False):

  num_args = len(sys.argv)

  if do_print:  
    print('arg_config()...')
    print("num args:",num_args,sys.argv[0])

  # always...
  parser = argparse.ArgumentParser(description="TF_Speech")

  # optional arguments
  parser.add_argument('--batches', nargs='*')
  parser.add_argument("--keras", help="use keras.utils.multi_gpu_model",action="store_true")
  parser.add_argument("--gpus", help="which GPUs to use", nargs='*')


  # always...
  args = parser.parse_args()

  if args.batches == None:
    sys.exit("You must specify at least one batch size. e.g. --batches 128")

  if args.gpus != None:
    if len(args.gpus) != len(args.batches):
      sys.exit("You must specify a batch size for each GPU. e.g. --gpus 0 1 --batches 128 64")
    # convert args.gpus to a list of integers
    for i,val in enumerate(args.gpus):
      args.gpus[i] = int(val)

  # convert args.batches to a list of integers
  for i,val in enumerate(args.batches):
    args.batches[i] = int(val)

  print(args.batches)  
  print(args.keras)  
  print(args.gpus)  

  return args

if __name__ == "__main__":
  
  print_devices()

  # parse the command line arguments
  args = arg_config()

  # get the mnist train data only  
  (x_train, y_train_1hot)  = gpu_maxing_model.get_MNIST_train_data()

  if args.gpus == None:
    # with no gpus specified use the default gpu
    model = gpu_maxing_model.get_model()
  elif len(args.gpus) == 1:
    # with one gpus specified use that gpu for all batches
    with tf.device('/gpu:'+str(args.gpus[0])):
      model = gpu_maxing_model.get_model()
  else:
    single_model = gpu_maxing_model.get_model()
    model = ratio_training_utils.multi_gpu_model(single_model, gpus=args.gpus, ratios=args.batches)
        
  model.compile(optimizer=optimizers.Adam(), 
                loss=losses.categorical_crossentropy)

  # the total batch size is the sum of the batch sizes for each GPU
  batch_size = sum(args.batches)

  # avoid using more than one epoch in a single training run
  n_training_steps = 60000 // batch_size - 1
  n_training_runs = 10
  
  print('number of training runs = ',n_training_runs)

  # get the first traing batch out of the way cause it always so slow!
  print("starting first batch")
  model.train_on_batch(x_train[0:batch_size],y_train_1hot[0:batch_size])
  print("Finished first batch.")
  print("Each of the {} training runs should take about 10 seconds...".format(n_training_runs))

  # now start the timer running
  start_time = time.time()
  last_time = start_time
  n_samples_trained = 0

  # ten training runs is all you should need
  for training_run in range(n_training_runs):

    # train train train...
    for i in range(1,(n_training_steps+1)*batch_size,batch_size):
      model.train_on_batch(x_train[i:i+batch_size],y_train_1hot[i:i+batch_size])
      n_samples_trained += batch_size
      # abort after 10 seconds
      if last_time + 10.0 < time.time():
        last_time = time.time()
        break

    time_took = time.time() - start_time
    sps = n_samples_trained / time_took
    
    if args.gpus == None:
      print('After {} samples default GPU: {:5.0f}sps'.format(n_samples_trained,sps))
    elif len(args.gpus) == 1:
      print('After {} samples GPU:{} {:5.0f}sps'.format(n_samples_trained,args.gpus[0],sps))
    else:
      print('After {} samples'.format(n_samples_trained))
      for gpu,batch in zip(args.gpus,args.batches):
        print('\t GPU:{}[{}] {:5.0f}sps'.format(gpu,batch,sps*batch/sum(args.batches)))
      print('\t Total:[{}] {:5.0f}sps'.format(batch_size,sps))
        
  print("Done!")
  