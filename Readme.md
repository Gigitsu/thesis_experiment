
# char-rnn pos tagger

## Requirements

This code is written in Lua and requires [Torch](http://torch.ch/). If you're on Ubuntu, installing Torch in your home directory may look something like:

```bash
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch;
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
```

See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using [LuaRocks](https://luarocks.org/) (which already came with the Torch install). In particular:

```bash
$ luarocks install luautf8
$ luarocks install nngraph
$ luarocks install optim
$ luarocks install nn
```

If you'd like to train on an NVIDIA GPU using CUDA (this can be to about 15x faster), you'll of course need the GPU, and you will have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Then get the `cutorch` and `cunn` packages:

```bash
$ luarocks install cutorch
$ luarocks install cunn
```

If you'd like to use OpenCL GPU instead (e.g. ATI cards), you will instead need to install the `cltorch` and `clnn` packages, and then use the option `-opencl` during training ([cltorch issues](https://github.com/hughperkins/cltorch/issues)):

```bash
$ luarocks install cltorch
$ luarocks install clnn
```

## Usage

### Data

All input data is stored inside the `torch_data/data` directory. You'll notice that the `torch_data/` directory does not exists, in fact that directory is created on first script run. All the needed data, except for the Evalita dataset, will be automatically downloaded in that folder for you.

All the checkpoints are stored in `torch_data/save` directory.

**Evalita**: In order to use the evalita dataset, you need to download the [evalita.tar.gz](https://www.dropbox.com/s/v521hxr43r7qmvc/evalita.tar.gz?dl=0) archive and unpack it in `torch_data/data/CoNLL` directory. At the end you should have `devel` and `train` files in `torch_data/data/CoNLL/Evalita`.

With
```bash
$ th init.lua
```
you can generate `torch_data/` directory and all the needed subdirectory without actually starting training.

### Training

To start training multiple rnn instances with one command you can use `create_experiment_start.lua` script.

Simply type
```bash
$ th create_experiment_starter.lua && ./experiment_starter.sh
```
to generate and execute a shell script with all the commands needed to start the experiment.

The script provides some parameter to customize the `experiment_starter.sh` generation. Below the full list of available parameters with default values:

```bash
$ th create_experiment_starter.lua -h
#Creates a shell script to start the experiment

#Options
#  -use_space       True to train with space too. [true]
#  -min_nodes       The min number of node to use. [128]
#  -max_nodes       The max number of node to use. [1024]
#  -max_layers      The max number of layers to use [5]
#  -min_seq_lengths The min value of sequence length to use [60]
#  -max_seq_lengths The max value of sequence length to use [100]
#  -seq_length_step The step value to use for increment sequence length [20]
#  -max_epochs      The max number of full passes through the training data [150]
```
