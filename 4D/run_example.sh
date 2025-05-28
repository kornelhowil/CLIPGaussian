conda activate dmiso
cd models/dmisomodel
export PYTHONPATH=.
python train.py -s ../../data/trex -m ../../output/trex --batch 4 -w --iterations 80000
python render.py  -m ../../output/trex

cd ..
cd ..
export PYTHONPATH=.
python train_style.py -s data/trex -m output_style/trex  --model_output output/trex --iterations 5000 --batch 4 --style_prompt "Wood" -w

cd models/dmisomodel
export PYTHONPATH=.
python render.py  -m ../../output_style/trex --iteration 5000