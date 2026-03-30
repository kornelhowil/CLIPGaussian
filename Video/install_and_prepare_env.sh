mkdir -p model
cd model
git clone https://github.com/gmum/VeGaS.git --recursive vegas
cd vegas
conda create -y -n vegas python=3.8
conda run -n vegas pip install submodules/diff-gaussian-rasterization
conda run -n vegas pip install submodules/simple-knn
conda run -n vegas pip install -r requirements.txt
cd ..
cd ..
conda run -n vegas pip install -r requirements.txt
