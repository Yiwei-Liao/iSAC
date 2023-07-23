import pandas as pd
import numpy as np
from scipy.stats import rayleigh

data = pd.read_csv('./embeddings dimension/FB15-237/entity/FB15k-237_10d.tsv', sep='\t')
embeddings = data.values

# Add Guassian Noise
guassian_noise = np.random.normal(0, 0.5, embeddings.shape)

noise_guassian_embed = embeddings+guassian_noise

# Add Rayleigh Noise
rayleigh_noise = rayleigh.rvs(scale=0.2, size=embeddings.shape)

noise_rayleigh_embed = embeddings+rayleigh_noise


