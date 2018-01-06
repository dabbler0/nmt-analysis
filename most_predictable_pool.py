import torch
import numpy
import json
from tqdm import tqdm
from numpy import newaxis as na
from torch.utils.serialization import load_lua
from itertools import product as p

languages = ['es', 'fr', 'ar', 'ru', 'zh']

all_networks = {}

for language, version in tqdm(p(languages, [1, 2, 3]), desc='loading', total=len(languages) * 3):
    network_name = 'en-%s-%d' % (language, version)

    # Load the description of the network
    # This will be a 4000x(sentence_length)x500 matrix.
    # Rehsape to be a (total_tokens)x500 matrix.
    all_networks[network_name] = torch.cat(load_lua(
        '../descriptions/%s.desc.t7' % (network_name,)
    )).cuda()

# Take 0, 1-moments
means = {}
stdevs = {}
for network in tqdm(all_networks, desc='mu, sigma'):
    means[network] = all_networks[network].mean(0, keepdim=True)
    stdevs[network] = (
        all_networks[network] - means[network].expand_as(all_networks[network])
    ).pow(2).mean(0, keepdim=True).pow(0.5)

errors = {network: {} for network in all_networks}

# Get all correlation pairs
for network, other_network in tqdm(p(all_networks, all_networks), desc='correlate', total=len(all_networks)**2):
    # Don't match within one network
    if network == other_network:
        continue

    # Try to predict this network given the other one
    X = (all_networks[other_network] - means[other_network]) / stdevs[other_network]
    Y = (all_networks[network] - means[network]) / stdevs[network]

    coefs = X.t().mm(X).inverse().mm(X.t()).mm(Y)
    prediction = X.mm(coefs)
    error = (prediction - Y).pow(2).mean(0).squeeze()

    errors[network][other_network] = error

all_neurons = []
# For each network, created an "annotated sort"
#
# Sort neurons by worst best correlation with another neuron
# in another network.
for network in tqdm(all_networks, desc='annotation'):
    for neuron in range(500):
        all_neurons.append(
            ('%s:%d' % (network, neuron), max(
                errors[network][other][neuron] for other in errors[network]
            ))
        )

all_neurons = sorted(all_neurons,
    key = lambda x: x[1]
)

json.dump(all_neurons, open('results/most-predictable-pool.json', 'w'), indent = 4)
