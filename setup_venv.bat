conda create --prefix=venv python=3.10.6  && ^
conda activate ./venv  && ^
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 && ^
pip install -r requirements.txt