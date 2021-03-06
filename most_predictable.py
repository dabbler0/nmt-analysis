import torch
import numpy
import json
from tqdm import tqdm
from numpy import newaxis as na
from torch.utils.serialization import load_lua
from itertools import product as p

languages = ['es', 'fr', 'ar', 'ru', 'zh']

all_networks = {}
position_labels = None

for language, version in tqdm(p(languages, [1, 2, 3]), desc='loading', total=len(languages) * 3):
    network_name = 'en-%s-%d' % (language, version)

    # Load the description of the network
    # This will be a 4000x(sentence_length)x500 matrix.
    # Rehsape to be a (total_tokens)x500 matrix.
    data = load_lua(
        '../descriptions/%s.desc.t7' % (network_name,)
    )

    all_networks[network_name] = torch.cat(data).cuda()

    # Create position annotations for each token
    if position_labels is None:
        position_labels = sum(
            (list(range(line.size()[0]))
            for line in data), []
        )

# Take 0, 1-moments
means = {}
stdevs = {}
for network in tqdm(all_networks, desc='mu, sigma'):
    means[network] = all_networks[network].mean(0, keepdim=True)
    stdevs[network] = (
        all_networks[network] - means[network].expand_as(all_networks[network])
    ).pow(2).mean(0, keepdim=True).pow(0.5)

    all_networks[network] = (all_networks[network] - means[network]) / stdevs[network]

errors = {network: {} for network in all_networks}

# Get all correlation pairs
for network, other_network in tqdm(p(all_networks, all_networks), desc='correlate', total=len(all_networks)**2):
    # Don't match within one network
    if network == other_network:
        continue

    # Try to predict this network given the other one
    X = all_networks[other_network].clone()
    Y = all_networks[network].clone()

    coefs = X.t().mm(X).inverse().mm(X.t()).mm(Y)
    prediction = X.mm(coefs)
    error = (prediction - Y).pow(2).mean(0).squeeze()

    errors[network][other_network] = error

neuron_notated_sort = {}

def get_position_mse(network):
    buckets = {}
    for i, position in enumerate(position_labels):
        if position not in buckets:
            buckets[position] = []

        buckets[position].append(all_networks[network][i])

    means = {x: torch.stack(buckets[x]).mean(0) for x in buckets}

    mse = torch.stack([
        v - means[position]
        for position in buckets
        for v in buckets[position]
    ]).pow(2).mean(0)

    return mse

position_errors = {
    network: get_position_mse(network)
    for network in tqdm(all_networks, desc='position')
}


# For each network, created an "annotated sort"
#
# Sort neurons by worst best correlation with another neuron
# in another network.
for network in tqdm(all_networks, desc='annotation'):
    neuron_sort = sorted(
        range(500),
        key = lambda i: max(
            errors[network][other][i]
            for other in errors[network] if other != 'position'
        )
    )

    # Annotate each neuron with its associated cluster
    neuron_notated_sort[network] = [
        (
            neuron,
            {
                other: errors[network][other][neuron]
                for other in errors[network]
            },
            position_errors[network][neuron]
        )
        for neuron in neuron_sort
    ]

json.dump(neuron_notated_sort, open('results/most-predictable.json', 'w'), indent = 4)
