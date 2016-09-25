This directory contains all the scripts that load/process datasets for respective models. 

The datasets used are:
- [MNIST](http://yann.lecun.com/exdb/mnist/), courtesy Yann Lecun. Download and extract files into `mnist` folder.
- [SVHN](http://ufldl.stanford.edu/housenumbers/), courtesy Netzer et al. Download **Format 1** train, test and extra datasets, and extract them to `svhn/train`, `svhn/test` and `svhn/extra` directories respectively. Additionally, convert the label `mat` files to v7 version. The dataset provides `mat` files stored in v7.3 version.