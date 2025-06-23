mkdir models
cd models
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive gs
cd gs
git checkout 54c035f7834b564019656c3e3fcc3646292f727d
conda env create --file environment.yml -n clipgaussian_3D
cd ..
cd ..
conda run -n clipgaussian_3D pip install -r requirements.txt
