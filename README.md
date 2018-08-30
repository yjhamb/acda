# ACDA
Attentive Contextual Denoising Autoencoder for Recommendation. Neural network architecture based on the Denoising Autoencoder that incorporated contextual data via the Attention Mechanism to provide personalized recommendation. The ACDA is a generic model that may be used for both rating prediction and top-N recommendation.

# Datasets
The model is evaluated against two datasets for the event recommendation 



# Execution
python3 latent_auto_encoder.py --epochs 100 --size 200 --corrupt 0.1 > ${base_dir}/result.txt &
