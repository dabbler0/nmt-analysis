from torch.utils.serialization import load_lua
import numpy
from numpy import newaxis as na
import json
from tqdm import tqdm
from itertools import product as p

languages = ['es', 'fr', 'ar', 'ru', 'zh']

all_networks = {}

for language, version in tqdm(p(languages, [1, 2, 3]), desc='loading', total=len(languages) * 3):
    network_name = 'en-%s-%d' % (language, version)

    # Load the description of the network
    # This will be a 4000x(sentence_length)x500 matrix.
    # Rehsape to be a (total_tokens)x500 matrix.
    all_networks[network_name] = numpy.concatenate([x.numpy() for x in load_lua(
        '../descriptions/%s.desc.t7' % (network_name,)
    )])

# Take 0, 1-moments
means = {}
stdevs = {}
for network in tqdm(all_networks, desc='mu, sigma'):
    means[network] = all_networks[network].mean(0)
    stdevs[network] = (
        (all_networks[network] - means[network][na, :]) ** 2
    ).mean(0) ** 0.5

correlations = {network: {} for network in all_networks}

# Get all correlation pairs
for network, other_network in tqdm(p(all_networks, all_networks), desc='correlate', total=len(all_networks)**2):
    # Don't match within one network
    if network == other_network:
        continue

    # Correlate these networks with each other
    covariance = (
        numpy.dot(
            all_networks[network].transpose(), all_networks[other_network] # E[ab]
        ) / all_networks[network].shape[1]
        - numpy.dot(
            means[network].transpose(), means[other_network] # E[a]E[b]
        )
    )

    correlation = covariance / numpy.dot(
        stdevs[network].transpose(), stdevs[other_network]
    )

    correlations[network][other_network] = correlation

# Get all "best correlation pairs"
clusters = {network: {} for network in all_networks}
for network, neuron in tqdm(p(all_networks, range(500)), desc='clusters', total=len(all_networks)*500):
    clusters[network][neuron] = {
        other: max(
            range(500),
            key = lambda i: abs(correlations[network][other][neuron][i])
        ) for other in correlations[network]
    }

neuron_notated_sort = {}
# For each network, created an "annotated sort"
#
# Sort neurons by worst best correlation with another neuron
# in another network.
for network in tqdm(all_networks, desc='annotation'):
    neuron_sort = sorted(
        range(500),
        key = lambda i: min(
            correlations[network][other][i][clusters[network][i][other]]
            for other in clusters[network][i]
        )
    )

    # Annotate each neuron with its associated cluster
    neuron_notated_sort[network] = [
        (
            neuron,
            {
                '%s:%d' % (other, clusters[network][neuron][other],):
                correlations[network][other][neuron][clusters[network][neuron][other]]
                for other in clusters[network][neuron]
            }
        )
        for neuron in neuron_sort
    ]

json.dump(neuron_notated_sort, open('results/annotated-clusters.json', 'w'))
