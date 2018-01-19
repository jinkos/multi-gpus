I am very proud of my two GPUs. One is a gtx1080 (8GB and fast) and the other a gtx1080 Ti (11GB and VERY fast).

I want to see them bleed.

Keras 2.0 comes with code that will distribute the the load evenly between two GPUs but this will see my 1080 Ti twiddling its thumbs while the 1080 is maxed out with it’s core temperature in the pushing 70.

In this repo are 3 files…

gpu_maxing_model.py contains a Keras MNIST model with more layers than it need. This should be able to get most GPUs up to 100% utilization provided than you are using a large enough batch size. Remember - If you are not maxing your batch size then you are not maxing your GPU.

ratio_training_utils.py contains my modified version of keras.utils.multi_gpu_model() that takes an extra parameter called ratios that takes a list of ratios for balancing the training load. For example ratio_training_utils.keras.utils.multi_gpu_model(model,gpus=2,ratios=[3,2]) will split the load roughly in the ratio 3:2 between the first two GPUs.

test_GPUs.py should be run from the command line and enables you to run the gpu_maxing_model on different GPUs with different ratios.

My suggestion for maximum fun is…
download the repo
read the simple instructions below so you can run test_GPUs.py
open a terminal and run watch -n 0.5 nvidia-smi so you can observe your GPU stats
open a second terminal for running test_GPUs.py
play with the parameters for test_GPUs.py until you have maxed out all your GPUs
use what you have learned to incorporate ratio_training_utils.py into your future Keras  

USING test_GPUs.py

python test_GPUs.py --batches=64

