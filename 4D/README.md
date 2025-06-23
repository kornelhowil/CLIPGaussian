<div align="center">
<h1> CLIPGaussian: for 4D scene</h1>
<div align="left">

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.22854-red)](https://arxiv.org/abs/2505.22854)  [![ProjectPage](https://img.shields.io/badge/Website-kornelhowil.github.io/CLIPGaussian/-blue)](https://kornelhowil.github.io/CLIPGaussian/) [![GitHub Repo stars](https://img.shields.io/github/stars/kornelhowil/CLIPGaussian.svg?style=social&label=Star&maxAge=60)](https://github.com/kornelhowil/CLIPGaussian)
</div>



### Requirements

- Conda
- CUDA toolkit 11 for PyTorch extensions (we used 11.8); see base model requirements, here [D-MiSo](https://github.com/waczjoan/D-MiSo)

## Clone the Repository

```shell
# SSH
git clone git@github.com:kornelhowil/CLIPGaussian.git
```
or
```shell
# HTTPS
git clone https://github.com/kornelhowil/CLIPGaussian.git
```

To install the required Python packages we used 3.8 python

## Fast start and train
To prepare repository and `dmiso` conda env, run:

```shell
sh install_and_prepare_env.sh
```

To run trex example:
```shell
sh run_example.sh #please check if paths are correct, sometimes full path is needed.
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

1. Train model 4D model (stage 1):

The first stage is to train the model reconstruction using a GS-based approach. For this, we selected [D-MiSo](https://github.com/waczjoan/D-MiSo) model 

We propose two setup scenarios:
- Option 1: Clone the base model repository first, then install our additional features. This approach allows you to customize your own model.
- Option 2: First clone CLIPGaussian repo, then clone your model.

In this tutorial, we’ll demonstrate the second approach.

Open command line in ..../CLIPGaussian/4D and run:
```shell
sh install_and_prepare_env.sh
```
It should clone D-MiSo model and create `dmiso` env using conda and additional install `requirements.txt`
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

2. Train style  based on 4D model: (stage 2)

We have: 
    - dataset (images and camera info in case of blender dataset) in `data/trex`
    - reconstruction object model created using Gaussian-Splatting based model D-MiSo in `output/trex`

We would like to created styled object trex by prompt `wood` using `output/trex`. New styled model will be saved in `output_style/trex`:

```shell
python train_style.py -s data/trex -m output_style/trex  --model_output output/trex --batch 4 --iterations 5000 --batch 4 --style_prompt "Wood" -w
```

you should find `output_style/trex_wood`: 
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

If you would like use image as a referenced style please use `--style_image path_to_image` instead of `style_prompt` 

5. Render styled images:
 
```shell
cd models/dmisomodel
export PYTHONPATH=.
python render.py  -m path/4D/output_style/trex_wood --iteration 5000 
 ```

In `output_style/trex_wood` you should find: 
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

If you find our work useful, please consider citing:
<h4 class="title">CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting

</h4>
    <pre><code>@Article{howil2025clipgaussian,
      author={Kornel Howil and Joanna Waczyńska and Piotr Borycki and Tadeusz Dziarmaga and Marcin Mazur and Przemysław Spurek},
      title={CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting},
      year={2025},
      eprint={2505.22854},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.22854}, 
}
</code></pre>

</div>

</section>

## Acknowledgments

The project “Effective rendering of 3D objects using Gaussian Splatting in an Augmented Reality environment” (FENG.02.02-IP.05-0114/23) is carried out within the First Team programme of the Foundation for Polish Science co-financed by the European Union under the European Funds for Smart Economy 2021-2027 (FENG).
<div align="center">
<img src="..//assets/fnp.png" />
</div>
