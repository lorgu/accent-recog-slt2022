wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
chmod +x Miniconda3-latest-Linux-x86_64.sh && \
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
export PATH="$HOME/miniconda/bin:$PATH" && \
conda create -n dicla python=3.10 -y && \
source $HOME/miniconda/bin/activate dicla && \
git clone https://github.com/lorgu/accent-recog-slt2022 /content/accent-recog-slt2022 && \
cd /content/accent-recog-slt2022/ && \
pip install -r requirements.txt && \
pip install torchtext==0.13.1 torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116