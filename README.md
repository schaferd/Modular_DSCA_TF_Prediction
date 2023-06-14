# Sparsely Connected Autoencoders for TF Activity Prediction

## Abstract:
The activities of transcription factors (TFs) control cellular activities and, in cases of
dysregulation, can cause disease. It is, however, challenging to measure TF activity experimentally, while gene expression data, 
its downstream product, can be readily obtained under a variety of conditions. We hypothesized that artificial neural network models could
be trained to predict TF activities from these data with the aid of prior knowledge networks
of TF to gene connections. Specifically, we expected that deep networks would capture
complex and non-linear relations between TFs and genes. Here we developed autoencoders
(AEs) of gene expression data with TF activities as the latent embedding trained on 36,794
samples. The AEs mimic TF-gene regulation by imposing network sparsity such that TF
nodes have a path to gene nodes only if the TF is known to regulate the gene. We evaluate
the performance of different architectures, using shallow sparse, deep-sparse, or fully connected modules for the encoder and decoder. 
We do a final comparison between the most and least constrained sparse models with a current standard method for predicting TF activities.
When compared to AEs with deep sparse modules, AEs with shallow sparse modules either
performed comparably or outperformed in tests to predict perturbed TFs and were more
consistent in tests to evaluate consistency across trained models.

## Installation
### Required Software:
### To train model (python3):
matplotlib            	3.5.2
numpy                 	1.22.4
pandas                	1.4.3
pyensembl             	2.0.0
pytorch               	1.10.2
scikit-learn          	1.1.1                	
scipy                 	1.7.3
seaborn               	0.11.2
sparselinear          	0.0.5
torch-scatter         	2.0.9
torch-sparse          	0.6.13
### To run DoRothEA comparison (R):
tidyverse   1.3.2
dorothea   1.6.0

## Download Data
Training Data: (url coming soon)
DoRothEA perturbation tests: (url coming soon)
Knock-out tests: (url coming soon)

Training or evaluating the model with your own data 	
Training or evaluation data must be a pandas DataFrame object with columns as Ensembl gene names and rows as samples. Training data should be saved as a pickle file. Data should be z-scored across genes. 

Train Model
`params.sh` contains a list of assignable hyperparameters and data filenames
`train.sh` contains the script to start the training
To train model using slurm run: `sbatch train.sh`

Evaluate Model
`eval_saved_models/params.sh` contains a list of assignable parameters and data filenames

`eval_saved_models/eval_models.py` requires that the list of run directories where model files can be found is assigned at the end of the script in the dictionary params by the key model_dirs. You must also specify which input data file(s) you are using as parameters to functions when calling evaluation methods toward the end of the script. Run using `sbatch eval_models.sh` while in the `eval_saved_models/` directory

`eval_saved_models/eval.py` requires that the input data file you are using be specified as parameters to the functions when calling evaluation methods (found towards the end of the script). Run using `sbatch eval.sh` while in the `eval_saved_models/` directory.

For both evaluation scripts: model module types in `params.sh` must match the module types in the saved model file. Other parameters like width and depth must also be identical. 

