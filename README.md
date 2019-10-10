#Mixnet-DSQ



## Introduction

Days have passed since I last submitted my submission.With the participation of ,we have found a better solution for the challenge.

We use the [**MixNet**](https://arxiv.org/abs/1907.09595) as the baseline model and apply the most recent quantization method on it. Finally ,our model achieves top-1 accuracy on Cifar100 with  a score of ,which is far better than my first submission.

## Requirement

+ Pytorch >1.1.0
+ Python >=3.6
+ [theconf](https://github.com/wbaek/theconf)
+ [tqdm](https://github.com/tqdm/tqdm)
+ [warmup_scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr)



## Install

`git clone `

''

## Detailed descriptions

In the following sections,we will explain our solution in details.

### Train from scratch

We train [**Mixnet**](https://arxiv.org/abs/1907.09595) from scratch on Cifar100 dataset.Our training code is referred to the implementation from [Fastautoaugment](https://github.com/kakaobrain/fast-autoaugment).However,in our training process,we do not use fast-autoaugment policy .

To further reduce the model complexity,we manually optimize the network structure.More specifically,we take  the following actions:

+ **stride changing**

  we change the stride from 4 to 2 to adapt the network to input image size 32.

+ **kernel removal**

  

+ **feature dimension reduction**

In addition ,we change the **depth_multiplier** to 0.83.

The original Mixnet

Besides,we use the following strategies to boost the performance of **Mixnet** on Cifar100 dataset.

+ [**Cutout**](https://arxiv.org/abs/1708.04552),[**reference implementation**](https://github.com/uoguelph-mlrg/Cutout)

  **Cutout** randomly cuts a patch out of a sample image to augment the dataset.Its hyper-parameter  *n_holes*  and  *length* are set to 1 and 16 separately in our experiment.

+ [**Mixup**](https://arxiv.org/abs/1710.09412),[**reference implementation**](https://github.com/facebookresearch/mixup-cifar10)

  **Mixup** adopts a convex combination of taining samples and their labels to improve  generalization. Its hyper-parameter *alpha* is set to 1.0 in our experiment.

+ [**warmup_scheduler**](https://arxiv.org/abs/1706.02677),[**reference implementation**](https://github.com/ildoonet/pytorch-gradual-warmup-lr)

  **warmup_scheduler** is to alleviate rapid changes in network at early training epochs.Its hype parameter *multiplier* and *warmup epoch* is set  to 1.01 and 10 in our experiment.

+ **LabelSmooth**

  To reguarize the training, **label smooth** is used where positive labels and negative labels are smoothed to  0.9 and 0.005 respectively.You can find the settings in *train.py* line 32~35.

Our  training settings are listed in the table below

| lr   | batchsize | 0ptimizer                                 | warm_up                     | lr_schedule | epoch |
| ---- | --------- | ----------------------------------------- | --------------------------- | ----------- | ----- |
| 0.1  | 768       | SGD with netsetrov<br>weight decay:0.0001 | Multiplier:1.01<br>epoch:10 | cosine<br>  | 500   |

You can find these settings in *mixnet_m.yaml*

With the above settings,our mixnet model achieves  **80.7%**  top-1 accuracy.



To reproduce the reported accuracy follow the steps below:

1. `cd FastAutoAugment`

2. In *run_train.sh*, specify *project_dir* , *dataset_dir* , *save* and *tag*

   *save* is the save model name

   *tag* is the name for the experiment

3. `./run_train.sh`



To evaluate the trained model

1. In *run_test.sh*, specify *project_dir* ,*dataset_dir*  *save* 

   *save* is the saved mdoel name above

2. `./run_test.sh`

The already trained model is available in [mixnet_google_drive](https://drive.google.com/open?id=1FvayLyx_KVDQeYV56lHMRE36lFCeFjba)

---



### DSQ

We quantize trained Mixnet to lower bits using [DSQ](https://arxiv.org/abs/1908.05033). You can find the implemntation details in *quantize_methods.py* and *quantize_methods.py* .According to the paper,we use the *soft quant* to quant and de-quant the tensors.The quant and de-quant operations are in *quantize_methods.py* line 54~109.

The convolution weight and input activations are quantized into 8bits and 8bits respectively.To guarantee the accuracy of quantized model,we exclude the first and last layers  in quantization.You can find these in *mixnet_dsq.py*.

Our quantization settings are listed in the table below

| weight bits | activation bits | weight alpha | activation alpha | per-channel      | Quant   activation | memo                           |
| ----------- | --------------- | ------------ | ---------------- | ---------------- | ------------------ | ------------------------------ |
| w_qbit=8    | act_qbit=8      | w_alpha=0.5  | act_alpha=0.5    | per_channel=True | act_quant=True     | Per_channel is for weight only |

You can find these variables in *DSQ_params* in *quantize_modules.py*

Our quantization fintuning settings are listed in the table below

| lr     | batchsize | optimizer                                  | warm_up                      | lr_schedule | epoch |
| ------ | --------- | ------------------------------------------ | ---------------------------- | ----------- | ----- |
| 0.0001 | 128       | SGD with netsetrov<br/>weight decay:0.0001 | Multiplier:1.01<br/>epoch:10 | Cosine      | 500   |

You can find these settings in *mixnet_m_quant.yml*

With the above settings,our quantized model still achieves top-1 accuracy on Cifar100 dataset.



To reproduce the reported accuracy follow the steps below:

1. `cd FastAutoAugment`

2. In *run_quant_train.sh*, specify *project_dir* , *dataset_dir* , *pretrained* ,*save*and *tag*

   *pretrained* is the trained full-precision model name 

   *tag* is the name for the experiment

   *save* is the quantized model name to be saved

3. `./run_quant_train.sh`



To evaluate the quantized model

1. In *run_quant_test.sh*, specify *project_dir* ,*dataset_dir*  *save* 

   *save* is the saved quantized mdoel name 

2. `./run_quant_test.sh`

The already trained model is available in [quantized_mixnet_google_drive]()





## Scoring



## Other

Our code is under MIT license.You can distribute our code .

Our team info  is as the following

Team name: **Sloth** 

Team members: 

**Xin Liu**  E-mail:750740751@qq.com

**Yao Zhang** E-mail:

The challenge can post our results under **Sloth ** on official website.



