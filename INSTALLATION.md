# Installation Instructions

1. Install anaconda on your computer - https://docs.anaconda.com/anaconda/install/

2. Create a new anaconda environment with python 3.7+
```bash
conda create -n deeplearning python=3.7
conda activate deeplearning
```

3. Check your CUDA version
```bash
nvcc --version
```
4. Install repository dependencies. For installing pytorch, check https://pytorch.org/ and match the install command with your CUDA version. The command below is for a computer with CUDA version 10 installed.
```bash
conda install matplotlib
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorboard
conda install -c conda-forge tensorboardX
conda install -c intel scikit-learn
pip install pyyaml
```

5. Install `deeplearning` repository
```bash
mkdir <PATH WHERE YOU WANT TO STORE THE REPOSITORY>
cd <PATH WHERE YOU WANT TO STORE THE REPOSITORY>
git clone -b https://github.com/zachares/deeplearning.git
cd supervised_learning
pip install -e .
cd ..
```
