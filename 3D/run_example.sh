# train model (stage 1)
python models/gs/train.py -s data/nerf_synthetic/lego -m output/lego --save_iterations 30_000
python models/gs/render.py  -m output/lego

# train objects using style (stage 2)
python train_style.py -s data/nerf_synthetic/lego -m output_style/lego_fire  --ply_path output/lego/point_cloud/iteration_30000/point_cloud.ply --iterations 5000 --style_prompt "Starry Night by Vincent van Gogh"

# render style
python models/gs/render.py -m output_style/lego_fire