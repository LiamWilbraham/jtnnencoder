# jtnnencoder
Pip installable version of the Junction Tree Variational Autoencoder https://arxiv.org/abs/1802.04364.
The original, full repository can be found here: https://github.com/wengong-jin/icml18-jtnn

Intended to be used as a simple interface for generating molecular features for molecules based on JTNN latent vectors.

## Install
```
$ pip install jtnnencoder
$ pip install torch
```

You will also need rdkit:
```
$ conda install -c rdkit rdkit
```

## Use

```
from jtnnencoder import JTNNEmbed

smiles = ['CC1(C)[C@H]2C[C@H](C/C=C\CCCC(=O)O)[C@@H](NC(=O)c3csc4ccc(O)cc34)[C@@H]1C2']
jtnn = JTNNEmbed(smiles)
features = jtnn.get_features()

```

`features` is then a numpy ndarray of dimension (n_smiles x n_features).