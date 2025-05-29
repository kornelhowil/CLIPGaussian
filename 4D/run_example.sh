conda activate dmiso

# train model (stage 1)
cd models/dmisomodel
export PYTHONPATH=.
python train.py -s ../../data/trex -m ../../output/trex --batch 8 -w --iterations 80000 --is_blender
python render.py  -m ../../output/trex

# train objects using style (stage 2)
cd ..
cd ..
export PYTHONPATH=.
python train_style.py -s data/trex -m output_style/trex  --model_output output/trex --iterations 5000 --batch 4 --style_prompt "Wood" -w

cd models/dmisomodel
export PYTHONPATH=.
python render.py  -m ../../output_style/trex_wood --iteration 5000