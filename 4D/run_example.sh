PYTHONPATH=
python models/D-MiSo/train.py -s data/trex -m output/trex --batch 4 -w


PYTHONPATH=
python train_style.py -s data/trex -m output_style/trex  --model_output output/trex --iterations 5000 --batch 4 --style_prompt "Wood" -w
