# Maxing Multiple GPUs of Different Sizes with Keras and TensorFlow

Keras 2.0 (w/ TF backend) provides support for multiple GPUs by allowing the GPU processing to be spread evenly among several GPUs.

Unfortunately if some GPUs are faster than others, the faster ones will only be given as much work as the slowest, leading to low utilization and sub-optimal performance.

This repo contains a modified version of `keras.utils.multi_gpu_model()` that can take an extra parameter which is a list of ratios denoting how the processing load should be split. e.g...
 
`multi_gpu_model(model,gpus=2,ratios=[4,3])` will spread the samples per batch in the ratio of 4:3 between GPU:0 and GPU:1

#### On this page

1. [If you are already using *keras.utils.multi\_gpu\_model()*](#if-you-are-already-using-kerasutilsmulti_gpu_model)
2. [Tutorial - How I Maxed out my 2 GPUs](#tutorial-how-i-maxed-out-my-2-gpus)
3. [Converting single GPU models to multi GPU models]()


## If you are already using *keras.utils.multi\_gpu\_model()*
You are 90% there. Download [ratio\_training\_utils.py](https://github.com/jinkos/multi-gpus/blob/master/ratio_training_utils.py) and replace you calls to `keras.utils.multi_gpu_model()` with equivalent calls to `ratio_training_utils.multi_gpu_model()`

Here are some quick usage examples...

`keras.utils.multi_gpu_model(model,gpus=2)`

`ratio_training_utils.multi_gpu_model(model,gpus=2)`

`ratio_training_utils.multi_gpu_model(model,gpus=2,ratios=[1,1])`

`ratio_training_utils.multi_gpu_model(model,gpus=2,ratios=[50,50])`

all do the same thing: on a per batch basis, they split the batches evenly between two GPUs. If the batch size is 128 then 64 will be given to each GPU and the results will be combined when both GPUs are finished.

`ratio_training_utils.multi_gpu_model(model,gpus=2,ratios=[768,560])`

is what I use to balance my gtx1080 and my gtx1080-Ti. If I was using a batch size of 100 then 58 of the 100 samples would be sent to the gtx1080-Ti and 42 would be sent to the gtx1080 (768:560 is almost 58:42).

`ratio_training_utils.multi_gpu_model(model,gpus=[0,1,2],ratios=[4,3,2])`

might work for a 3 GPU system

IMPORTANT: Always use the largest batch size that your GPU can handle or you won't be using its full potential. Keep increasing the batch size until you start to get memory warning. Similarly, if you are using 4 identical GPUs then you should increase you overall batch size to four times what it was on a single GPU. See the Turorial for a practical example.

## Tutorial: How I Maxed out my 2 GPUs

I am very proud of my two GPUs. One is a gtx1080 (8GB and fast) and the other a gtx1080 Ti (11GB and VERY fast).

I want to see them bleed.

Keras 2.0 comes with code that will distribute the the load evenly between two GPUs but this will see my 1080 Ti twiddling its thumbs while the 1080 is maxed out with it’s core temperature in the low 70s.

In this repo are 3 files…

[gpu\_maxing\_model.py](https://github.com/jinkos/multi-gpus/blob/master/gpu_maxing_model.py) contains a Keras MNIST model with FAR more layers than it needs. (Please don't try and make it converge - that's not what it's for). This model should be able to get most GPUs up to 100% utilization provided than you are using a large enough batch size. Remember - If you are not maxing your batch size then you are not maxing your GPU.

[ratio\_training\_utils.py](https://github.com/jinkos/multi-gpus/blob/master/ratio_training_utils.py) contains my modified version of `keras.utils.multi_gpu_model()` that takes an extra parameter: a list of ratios for balancing the training load. For example:

`ratio_training_utils.keras.utils.multi_gpu_model(model,gpus=2,ratios=[3,2])` 

will split the load roughly in the ratio 3:2 between the first two GPUs.

[test\_GPUs.py](https://github.com/jinkos/multi-gpus/blob/master/test_GPUs.py) should be run from the command line and enables you to run the `gpu_maxing_model` on different GPUs with different ratios.

I download the 3 files into a directory called `let_them_bleed` and I'm ready to roll.

I open a terminal and type:

`watch -n0.5 nvidia-smi`

so that I can observe my GPUs' utilization and temperature. I open a second terminal and type:

`python3 test_GPUs.py --batches 64`

after about 30 seconds my output looks like this...

```
Each of the 10 training runs should take about 10 seconds...
After 6848 samples default GPU:   683sps
After 13760 samples default GPU:   684sps
After 20608 samples default GPU:   684sps
```

which is telling me that my default GPU is running at 684sps (samples per second). The GPU watcher is saying that my gtx1080-Ti is running at 96% which, it turns out, is a lie!

I quadruple the batch size:

`python3 test_GPUs.py --batches 256`

producing:

```
Each of the 10 training runs should take about 10 seconds...
After 11520 samples default GPU:  1128sps
After 23040 samples default GPU:  1129sps
After 34560 samples default GPU:  1129sps
```

It's almost doubled the sps! My GPU temp is up to 55C. Too cold...

`python3 test_GPUs.py --batches 1024`

producing:

```
Each of the 10 training runs should take about 10 seconds...
After 14336 samples default GPU:  1346sps
After 28672 samples default GPU:  1344sps
After 43008 samples default GPU:  1345sps
```

a temp of 62C and a utilization of 100%. I can't get the batch sizes any larger because I start getting memory warnings. Now it's time for the second GPU. I run the following:

`python3 test_GPUs.py --gpus 1 --batches 512`

producing:

```
Each of the 10 training runs should take about 10 seconds...
After 9216 samples GPU:1   878sps
After 18432 samples GPU:1   875sps
After 27648 samples GPU:1   873sps
```

Which is not as fast as the 1080-Ti. Now let's run both together...

`python3 test_GPUs.py --gpus 0 1 --batches 512 512`

This runs 512 batches on both GPUs at the same time and produces the following output:

```
Each of the 10 training runs should take about 10 seconds...
After 18432 samples
	 GPU:0[512]   901sps
	 GPU:1[512]   901sps
	 Total:  1801sps
After 36864 samples
	 GPU:0[512]   906sps
	 GPU:1[512]   906sps
	 Total:  1811sps
After 55296 samples
	 GPU:0[512]   905sps
	 GPU:1[512]   905sps
	 Total:  1810sps
```

Clearly both GPUs are doing the same amount of work, but the GPU watcher is telling me that the gtx1080 is maxed-out and the gtx1080-Ti is running at around 60%. It turns out that with a batch size of 512, on it's own, the 1080-Ti will manage 1260sps while the 1080 will manage 875. So that's the sort of ratio that I need to use to balance the load. Let's say I was running a 512 batch size on GPU:0 what would I need on GPU:1? I guess 512 * 875 / 1260 = 355. Let's try it...

`python3 test_GPUs.py --gpus 0 1 --batches 512 355`

produces:

```
Each of the 10 training runs should take about 10 seconds...
After 20808 samples
	 GPU:0[512]  1196sps
	 GPU:1[355]   829sps
	 Total:  2025sps
After 41616 samples
	 GPU:0[512]  1197sps
	 GPU:1[355]   830sps
	 Total:  2026sps
After 62424 samples
	 GPU:0[512]  1199sps
	 GPU:1[355]   831sps
	 Total:  2030sps
```
	 
Not bad, we're beating the total sps we had when we were evenly balancing the GPUs. But I think we can do better. Let's double the batch sizes...

`python3 test_GPUs.py --gpus 0 1 --batches 1024 710`

Dang!! I'm getting 'out of memory errors'. It was bound to happen, eventually. Everything down by 20%...

`python3 test_GPUs.py --gpus 0 1 --batches 820 568`

That's more like it...

```
Each of the 10 training runs should take about 10 seconds...
After 22208 samples
	 GPU:0[820]  1257sps
	 GPU:1[568]   871sps
	 Total:  2127sps
After 44416 samples
	 GPU:0[820]  1254sps
	 GPU:1[568]   868sps
	 Total:  2122sps
After 66624 samples
	 GPU:0[820]  1255sps
	 GPU:1[568]   869sps
	 Total:  2123sps
```
	 
Both GPUs are looking pretty maxed out and the termperature on the gtx1080 is in the high 60s. 2100sps is about 16% higher than the 1800sps that I was getting when the loads were balanced evenly.

So, on our journey from 684sps to 2123sps what have we learned?

Firstly we have learned that we should be using large batch sizes. That's a good rule even if you are just using one GPU.

Second, in this particular case, we need to be using a ratio something like 820:568 for balancing between my particular GPUs.

Thirdly, I hope you have learned how you can use the code in the repo to find the right balance for whatever your GPU combo is.

At last I need to address the question of how to use 
