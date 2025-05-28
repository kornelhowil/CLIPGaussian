mkdir models
cd models
git clone git@github.com:waczjoan/D-MiSo.git --recursive  dmisomodel
cd dmisomodel
sh install.sh
cd ..
cd ..
conda run -n dmiso pip install -r requirements.txt
