# Instruction of train.py

## Training Acceleration

### Prefetch dataset

*Default:* Open, use --no-prefetcher to disable

Prefetch dataset will prefetch a batch and copy the data to device of each process

### Multi-epoch dataloader

*Default:* Closed, use --use-multi-epochs-loader to enable

With multi-epoch dataloader, the iterator of the dataloader will not change in different epochs which will accelerate the training process a the beginning of each epoch.

### Pinned Memory

*Default:* Closed, use --pin-mem to enable

Use pinned memory in dataloader.

### AMP acceleration

*Default:* Closed, use --amp to enable, you set the opt_level in the training code.

AMP is an open source project of NVIDIA, it wraps the Distributed Data Parallel of PyTorch and optimize the backend communication. Meanwhile, the library has mixed precision optimization from O0 (pure FP32）to O3 （Pure FP16), more details can be found in https://nvidia.github.io/apex/amp.html

### Tar Dataset

*Default:* Closed, modifiy the training code to enable

Timm offers a dataset class which can directly set up a dataset from a tar file. The class is defined in timm/data/dataset.py#L129

## Data Augmentation

### Multi-split

*Default*：Closed, use --aug-splits split_num(int) to enable, split_num should be bigger than 1

With mulit-split, you can get split times of the number of training data in an epoch. One split will just have normalization and the other splits will go through normalization and data augmentation transforms like rotation and random crop.

### Mix-up

*Default:* Closed, use --mixup alpha(float) to enable

mixup is happend in the data loader, the i-th item in the batch will be alpha*batch[i]+(1-alpha)*batch[batchsize-i]

### Label smoothing

*Default:* Enabled, use --smoothing alpha(float) to adjust

## Training Optimization

### Sync-bn

*Default:* Closed, use --sync-bn to enable

the parameter of BN layer will be synchronized at the end of each epoch.

### Model-EMA

*Default:* Closed, use --model-ema to enable

Model-ema will calculate the weight of model like $\theta_i = \alpha \theta_{i-1}+ (1-\alpha) g_i$, which will help with the robustness of the model.



