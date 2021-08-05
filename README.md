# Deep Learning Models for bone suppression in chest radiographs
 Gusarev's bone suppression method, as described in https://doi.org/10.1109/CIBCB.2017.8058543.
 
 The software here has been implemented in Python 3.6 and PyTorch.  A batch normalisation layer was added in the convolutional blocks order to improve image quality and reduce artefacts.
Hence, convolutional blocks now look like: conv->batchnorm->relu
 
 # How to run the software using the pre-trained model for evaluation and/or testing:
 The pre-trained model is in the root directory, called "trained_network.tar". It is trained on 256x256 PNG images.
 This tar file is a dictionary containing the model state-dict, optimiser state-dict, and scheduler state-dict, the number of epochs completed and the number of real image-pairs shown to train the model.
 
 1) Input your desired dataset into datasets.py

Then:

 2) Run analysis_script.ipynb on jupyter notebook or jupyter lab


 
 # Training dataset used to generate the pre-trained model:
 Training data from the JSRT Chest X-ray dataset (not included) by Shiraishi et al. (https://doi.org/10.2214/ajr.174.1.1740071) and the bone-suppressed version by Juhász et al. (https://doi.org/10.1007/978-3-642-13039-7_90).  The dataset can be downloaded from https://www.kaggle.com/hmchuong/xray-bone-shadow-supression.
 
 # To train on your own data:
 Modify the main.ipynb file as needed in order to implement the correct settings and paths.
 Modify the datasets.py file to cleanly implement custom datasets.
 
 # Other people's software
 Pytorch-MSSSIM from https://github.com/jorge-pessoa/pytorch-msssim and was used for training, as specified in the Gusarev et al. paper.
 torchviz is from https://github.com/szagoruyko/pytorchviz/tree/master/torchviz and was used for debugging.
 
 
