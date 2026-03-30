<div align="center">
<h1> CLIPGaussian: for Video</h1>
<div align="left">

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.22854-red)](https://arxiv.org/abs/2505.22854)  [![ProjectPage](https://img.shields.io/badge/Website-kornelhowil.github.io/CLIPGaussian/-blue)](https://kornelhowil.github.io/CLIPGaussian/) [![GitHub Repo stars](https://img.shields.io/github/stars/kornelhowil/CLIPGaussian.svg?style=social&label=Star&maxAge=60)](https://github.com/kornelhowil/CLIPGaussian)
</div>

### Requirements

- Conda
- CUDA toolkit 12 for PyTorch extensions (see base model requirements [VeGaS](https://github.com/gmum/VeGaS))

## Clone the Repository

To install the required Python packages we used Python 3.8

## Fast start and train
To prepare repository and environment, run:

```shell
bash install_and_prepare_env.sh
```

## Tutorial 
In this section we describe how to train and render stylized videos using **VeGaS** as the base representation.

1. Train base video model (stage 1):

The first stage is to train the video reconstruction using [VeGaS](https://github.com/gmum/VeGaS).

Before training, your video needs to be converted to individual frames (0000.png, 0001.png, ...).
The data directory needs to have a structure like this:
```
<dataset_dir>
|---original
|   |---0000.png
|   |---0001.png
|   |---...
```

Run Stage 1 training:
```shell
python models/vegas/train.py -s <dataset_dir> -m output/base_video --save_iterations 30_000
```

2. Train style based on video model: (stage 2)

We would like to create a stylized video using a prompt (e.g., "Starry Night") based on the reconstruction from Stage 1.

```shell
python train_style.py -s <dataset_dir> -m output_style/video_stylized --ply_path output/base_video/point_cloud/iteration_30000/point_cloud.ply --iterations 5000 --style_prompt "Starry Night by Vincent van Gogh"
```

If you would like to use an image as a reference style, use `--style_image path_to_image` instead of `style_prompt`.

3. Render stylized video:
 
```shell
python models/vegas/render.py -m output_style/video_stylized
```

If you find our work useful, please consider citing:
<h4 class="title">CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting</h4>
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
