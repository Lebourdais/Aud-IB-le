# Aud-IB-le
Repository for experiments on explainable models for audio processing. 

### Sparse Autoencoders for Audio Foundation Models Explainability

This repository hosts the code for the following paper:
```
@article{mariotte2025sparse,
  title={Sparse Autoencoders Make Audio Foundation Models more Explainable},
  author={Mariotte, Th{\'e}o and Lebourdais, Martin and Almud{\'e}var, Antonio and Tahon, Marie and Ortega, Alfonso and Dugu{\'e}, Nicolas},
  journal={arXiv preprint arXiv:2509.24793},
  year={2025}
}
```

The code to reproduce the experiments performed in the paper is available in the following folders:
- `sae_ssl` : scripts to run the training of probes and SAEs in pretrained audio models
- `nn` : contains the code with pytorch models. This paper used only the `pretrained_model.py` and `sae.py` files
- `data` : contain the VocalSet dataloader. Also a dataloader for ESC50 that was used to conduct some experiments that are not reported in the paper.
- `attributes` : code to process EGMAPS features after being extracted. 
- `eval`: contain the disentanglement metrics and the notebooks to evaluate the models in terms of classification accuracy, reconstruction capacity of the SAE, disentanglement and identifiability (mapping dimensions from features).

