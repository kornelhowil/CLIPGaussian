mkdir models
cd models
git clone git@github.com:waczjoan/D-MiSo.git --recursive
git checkout rasterizer

sh install.sh

cd ..

pip install -r requirements.txt
