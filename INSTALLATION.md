1. Install anaconda on your computer - https://docs.anaconda.com/anaconda/install/

2. Create a new anaconda environment with python 3.7+
    conda create -n deeplearning python=3.7
    conda activate deeplearning

3. Install all repository dependencies
    Check your CUDA version to make sure you install the right pytorch libraries
    nvcc --version

    Install repository dependencies
    conda install matplotlib
    conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
    conda install -c conda-forge tensorboard
    conda install -c conda-forge tensorboardX
    conda install -c intel scikit-learn
    pip install pyyaml

4. Install deeplearning repository
    mkdir <<PATH WHERE YOU WANT TO STORE THE REPOSITORY>>
    cd <<PATH WHERE YOU WANT TO STORE THE REPOSITORY>>
    git clone -b https://github.com/zachares/supervised_learning.git
    cd supervised_learning
    pip install -e .
    cd ..