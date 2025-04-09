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

## Tutorial 
In this section we describe more details, and make step by step how to train and render 4D Gaussian Splatting model (we use `D-MiSo`) using style.
1. Go to [D-NeRF Datasets](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=2&dl=0), download `mutant` dataset and put it in to `data` directory. For example:

```
<D-MiSo>
|---submodules
|   |---<D-MiSo>
|---data
|   |---<mutant>
|   |---<jumpingjacks>
|   |---...
|---train.py
|---metrics.py
|---...
```

2. Train model 4D model:

  ```shell
cd submodules/D-MiSo 
python submodules/D-MiSo/train.py -s "../../data/mutant" -m "../../output/mutant" --iterations 80000 
  --warm_up 2000  --densify_until_iter 5000   
  --num_gauss 100000 --num_splat 25 --batch_size 10 -r 2 --is_blender
  ```
Tip1: use `-w` if you want white background
Tip2: ignore `-r 2` if you want resolution=1 (longer training time)

In `output/jumpingjacks` you should find: 
```
<4D>
|---data
|   |---<mutant>
|   |   |---transforms_train.json
|   |   |---...
|---output
|   |---<mutant>
|   |   |---deform
|   |   |---time_net
|   |   |---point_cloud
|   |   |---xyz
|   |   |---cfg_args
|   |   |---...
|---...
```

3. Train style based on 4D model:

  ```shell
cd ../..
python train_styles.py -m "output_style/mutant" --model_output "output/mutant" --iterations 5000 --batch_size 4 
  --style_prompt "Fire" --feature_lr 0.0025 --crop_size 64 --num_crops 64 
 ```

you should find `output_style/mutant` you should find: 
```
<4D>
|---data
|   |---<mutant>
|   |   |---transforms_train.json
|   |   |---...
|---output
|   |---<mutant>
|   |   |---...
|---output_style
|   |---<mutant>
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
 python render.py -m output_style/mutant 
  ```
In `output/jumpingjacks` you should find: 
```
<4D>
|---data
|   |---...
|---output
|   |---...
|---output_style
|   |---<mutant>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---train
|   |   |   |---<ours_best>
|   |   |---results.json
|   |   |---...
|---...
```