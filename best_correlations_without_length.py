import torch
import numpy
import json
from tqdm import tqdm
from numpy import newaxis as na
from torch.utils.serialization import load_lua
from itertools import product as p

languages = ['es', 'fr', 'ar', 'ru', 'zh']

all_networks = {}
all_networks_uncat = {}
for language, version in tqdm(p(languages, [1, 2, 3]), desc='loading', total=len(languages) * 3):
    network_name = 'en-%s-%d' % (language, version)

    # Load the description of the network
    # This will be a 4000x(sentence_length)x500 matrix.
    # Rehsape to be a (total_tokens)x500 matrix.
    all_networks[network_name] = [x.float().cuda() for x in load_lua(
        '../descriptions/%s.desc.t7' % (network_name,)
    )]

# Subtract out information that can be gleaned just
# by looking at position.
for network in tqdm(all_networks, desc='pos info'):
    # Sentence lengths:
    lengths = [x.size()[0] for x in all_networks[network]]

    # Turn the sequences into a padded sequence
    sequences = torch.zeros(len(all_networks[network]), max(lengths), 500).float().cuda()
    for i, line in enumerate(all_networks[network]):
        sequences[i, :lengths[i], :] = line

    # Create mask tensor, which will encode lengths simultaneously
    length_tensor = torch.Tensor(lengths).view(-1, 1, 1).expand_as(sequences)
    mask = torch.arange(0, sequences.size()[1]).view(1, -1, 1).expand_as(sequences)
    mask = mask.lt(length_tensor).float().cuda()

    # samples x seq_len x embedding_size
    # Take position-wise mean across samples
    # This will be (seq_len x embedding_size)
    pos_mean = (sequences.sum(0) / mask.sum(0)).unsqueeze(0).expand_as(sequences)

    # Subtract out position-wise means and concatenate
    all_networks[network] = torch.cat(all_networks[network]) - torch.cat([
        pos_mean[i, :l, :]
        for i, l in enumerate(lengths)
    ])

# Take 1, 2-moments
means = {}
stdevs = {}
for network in tqdm(all_networks, desc='mu, sigma'):
    means[network] = all_networks[network].mean(0, keepdim=True)
    stdevs[network] = all_networks[network].std(0, keepdim=True)

correlations = {network: {} for network in all_networks}

# Get all correlation pairs
for network, other_network in tqdm(p(all_networks, all_networks), desc='correlate', total=len(all_networks)**2):
    # Don't match within one network
    if network == other_network:
        continue

    # Correlate these networks with each other
    covariance = (
        torch.mm(
            all_networks[network].t(), all_networks[other_network] # E[ab]
        ) / all_networks[network].size()[0]
        - torch.mm(
            means[network].t(), means[other_network] # E[a]E[b]
        )
    )

    correlation = covariance / torch.mm(
        stdevs[network].t(), stdevs[other_network]
    )

    correlations[network][other_network] = correlation.double().cpu().numpy()

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
        key = lambda i: -min(
            abs(correlations[network][other][i][clusters[network][i][other]])
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

json.dump(neuron_notated_sort, open('results/annotated-clusters-no-length.json', 'w'), indent = 4)
