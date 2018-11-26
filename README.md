# ACDA
Attentive Contextual Denoising Autoencoder (ACDA) for top-N recommendation. The neural network architecture is based on the denoising autoencoder that incorporates contextual data via the attention mechanism to provide personalized recommendation. 
ACDA is a generic model that may be used for both rating prediction and top-N recommendation.

# Architecture

The neural network architecture is provided below. There is one hidden layer and the context is applied to the hidden representation via the attention mechanism.
The model is flexible enough to accommodate any number of contextual parameters; however, only two contextual parameters are depicted in the diagram for reference.

![ACDA Architecture](./acda-model.pdf) 

# Datasets
The model is evaluated against two datasets for the event recommendation 



# Execution
python3 latent_auto_encoder.py --epochs 100 --size 200 --corrupt 0.1 > ${base_dir}/result.txt &
