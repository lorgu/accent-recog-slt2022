wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
chmod +x Miniconda3-latest-Linux-x86_64.sh && \
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
export PATH="$HOME/miniconda/bin:$PATH" && \
conda create -n dicla python=3.10 -y && \
source $HOME/miniconda/bin/activate dicla && \
mkdir /nas && \
mkdir /nas/projects && \
mkdir /nas/projects/vokquant && \
mkdir /nas/projects/vokquant/data && \
mkdir /nas/projects/vokquant/data/dicla && \
git clone https://github.com/lorgu/accent-recog-slt2022 /nas/projects/vokquant/accent-recog-slt2022 && \
cd /nas/projects/vokquant/accent-recog-slt2022/ && \
pip install -r requirements.txt && \
pip install torchtext==0.13.1 torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 && \
apt-get update && apt-get install -y pigz && \
tar --use-compress-program=pigz -xf /workspace/processed_16khz_renamed.tar.gz -C /nas/projects/vokquant/data/dicla/
