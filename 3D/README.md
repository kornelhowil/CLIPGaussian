<div align="center">
<h1> CLIPGaussian: for 3D objects</h1>
<div align="left">

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.22854-red)](https://arxiv.org/abs/2505.22854)  [![ProjectPage](https://img.shields.io/badge/Website-kornelhowil.github.io/CLIPGaussian/-blue)](https://kornelhowil.github.io/CLIPGaussian/) [![GitHub Repo stars](https://img.shields.io/github/stars/kornelhowil/CLIPGaussian.svg?style=social&label=Star&maxAge=60)](https://github.com/kornelhowil/CLIPGaussian)
</div>



### Requirements

- Conda
- CUDA toolkit 11 for PyTorch extensions (we used 11.8); see base model requirements, here [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

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
To prepare repository and `gaussian_splatting` conda env, run:

```shell
bash install_and_prepare_env.sh
```

If you have `gaussian_splatting` conda env already installed you can just run 
```shell
conda run -n gaussian_splatting pip install -r requirements.txt
```

To download `nerf_sythetic` dataset run
```shell
bash download_data.sh
```

To run lego example:
```shell
bash run_example.sh #please check if paths are correct, sometimes full path is needed.
```

## Tutorial 
Download `nerf_sythetic` dataset using script
```shell
bash download_data.sh
```
or directly from the [Google Drive](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4). Your file sctructure should look like this

```
<CLIPGaussian>
|---3D
|   |---data
|   |   |---nerf_synthetic
|   |   |   |---<lego>
|   |   |   |---<hotdog>
|   |   |   |---...
|---train_style.py
...
```

1. Train model 4D model (stage 1):

The first stage is to train the model reconstruction using a GS-based approach. For this, we selected original [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) model.

We propose two setup scenarios:
- Option 1: Clone the base model repository first, then install our additional features. This approach allows you to customize your own model.
- Option 2: First clone CLIPGaussian repo, then clone your model.

In this tutorial, we’ll demonstrate the second approach.

Open command line in ..../CLIPGaussians/3D and run:
```shell
bash install_and_prepare_env.sh
```
It should clone gaussian-splatting model and create `gaussian_splatting` env using conda and additional install `requirements.txt`
```
<3D>
|---data
|   |---nerf_synthetic
|   |   |---<lego>
|   |   |---<hotdog>
|   |   |---...
|---models
|   |---<gs>
|---train_style.py
...
```

Train gaussian-splatting model to create reconstruction of `lego` object.

```shell
python models/gs/train.py -s data/nerf_synthetic/lego -m output/lego --save_iterations 30_000
 ```
Tip: use `-w` if you want white background

In `output/lego` you should find: 
```
<3D>
|---data
|   |---nerf_synthetic
|   |   |---<lego>
|   |   |---<hotdog>
|   |   |---...
|---output
|   |---<lego>
|   |   |---point_clound
|   |   |   |---iteration_30000
|   |   |   |   |---<point_clound.ply>
|   |   |---...
|---models
|   |---<gs>
|---train_style.py
...
```

2. Train style based on 3D model: (stage 2)

We have: 
    - dataset (images and camera info in case of blender dataset) in `data/nerf_synthetic/lego`
    - reconstruction object model created using Gaussian-Splatting based model in `output/lego`

We would like to created styled object trex by prompt `fire` using `output/lego`. New styled model will be saved in `output_style/lego_fire`:

```shell
python train_style.py -s data/nerf_synthetic/lego -m output_style/lego_fire  --ply_path output/lego/point_cloud/iteration_30000/point_cloud.ply --iterations 5000 --style_prompt "Fire"
```

you should find `output_style/lego_fire`: 
```
<3D>
|---data
|   |---nerf_synthetic
|   |   |---<lego>
|   |   |---<hotdog>
|   |   |---...
|---output
|   |---<lego>
|   |   |---point_clound
|   |   |   |---iteration_30000
|   |   |   |   |---<point_clound.ply>
|   |   |---...
|---output_style
|   |---<lego_fire>
|   |   |---point_clound
|   |   |   |---iteration_30000
|   |   |   |   |---<point_clound.ply>
|   |   |---...
|---models
|   |---<gs>
|---train_style.py
...
```

If you would like use image as a referenced style please use `--style_image path_to_image` instead of `style_prompt` 

5. Render styled images:
 
```shell
python models/gs/render.py -m output_style/lego_fire
```

In `output/lego` you should find: 
```
<3D>
|---data
|   |---...
|---output
|   |---...
|---output_style
|   |---<lego_fire>
|   |   |---point_cloud
|   |   |---train
|   |   |   |---<5000>
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
<img src="../assets/fnp.png" />
</div>
