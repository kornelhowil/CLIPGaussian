<div align="center">
<h1> CLIPGaussians: for 4D scene</h1>

<div align="left">

# Installation

TODO

### Requirements

- Conda (recommended)
- CUDA toolkit 11 for PyTorch extensions (we used 11.8)

## Clone the Repository with submodules

```shell
# SSH
git clone git@github.com:kornelhowil/CLIPGaussians.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/kornelhowil/CLIPGaussians.git --recursive
```

To install the required Python packages we used 3.8 python 

TODO

## Fast start and train
To prepare repository and `dmiso` conda env, run:

```shell
sh install_and_prepare_env.sh
```

To run trex example:
```shell
sh run_example.sh
```

## Tutorial 
In this section we describe more details, and make step by step how to train and render 4D Gaussian Splatting model (we use `D-MiSo`) using style.
Go to [D-NeRF Datasets](https://www.albertpumarola.com/research/D-NeRF/index.html), download `trex` dataset and put it in to `data` directory. For example:

```
<CLIPGaussian>
|---4D
|   |---data
|   |   |---<trex>
|   |   |---<jumpingjacks>
|   |   |---...
|---train_style.py
...
```

1. Train model 4D model:

The first stage is to train the model reconstruction using a GS-based approach. For this, we selected [D-MiSo](https://github.com/waczjoan/D-MiSo) model 

We propose two setup scenarios:
- Option 1: Clone the base model repository first, then install our additional features. This approach allows you to customize your own model.
- Option 2: First clone CLIPGaussian repo, then clone your model.

In this tutorial, we’ll demonstrate the second approach.

Open command line in ..../CLIPGaussians/4D and run:
```shell
sh install_and_prepare_env.sh
```
It should clone D-MiSo model and create `dmiso` env using conda.
```
<CLIPGaussian>
|---4D
|   |---data
|   |   |---<trex>
|   |   |---<jumpingjacks>
|   |   |---...
|   |---models
|   |   |---<dmisomodel>
|---train_style.py
...
```

Train D-MiSo model to create reconstruction of `trex` object.

```shell
cd models/dmisomodel
export PYTHONPATH=.
python train.py -s ../../data/trex -m ../../output/trex --batch 8 -w --iterations 80000 --is_blender
 ```
Tip1: use `-w` if you want white background
Tip2: ignore `-r 2` if you want resolution=1 (longer training time)

In `output/trex` you should find: 
```
<CLIPGaussian>
|---4D
|   |---data
|   |   |---<trex>
|   |   |---<jumpingjacks>
|   |   |---...
|   |---output
|   |   |---<trex>
|   |   |   |---deform
|   |   |   |---time_net
|   |   |   |---point_cloud
|   |   |   |---xyz
|   |   |   |---cfg_args
|   |   |   |---...
|   |---models
|   |   |---<dmisomodel>
|---train_style.py
...
```

2. Train style  based on 4D model:



you should find `output_style/trex_wood` you should find: 
```
<4D>
|---data
|   |---<trex>
|   |   |---transforms_train.json
|   |   |---...
|---output
|   |---<trex>
|   |   |---...
|---output_style
|   |---<trex>
|   |   |---deform
|   |   |---time_net
|   |   |---point_cloud
|   |   |---xyz
|   |   |---cfg_args
|   |   |---...
|---...
```

4. Render styled images:
 
```shell
cd models/dmisomodel
export PYTHONPATH=.
python render.py  -m path/4D/output_style/trex_wood --iteration 5000
 ```

In `output/jumpingjacks` you should find: 
```
<4D>
|---data
|   |---...
|---output
|   |---...
|---output_style
|   |---<trex_wood>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---train
|   |   |   |---<5000>
|   |   |---results.json
|   |   |---...
|---...
```