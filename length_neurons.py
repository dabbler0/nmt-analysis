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
    all_networks[network_name] = [x.cuda() for x in load_lua(
        '../descriptions/%s.desc.t7' % (network_name,)
    )]

# Take 1, 2-moments
means = {}
stdevs = {}
for network in tqdm(all_networks, desc='mu, sigma'):
    tensor = torch.cat(all_networks[network])
    means[network] = tensor.mean(0)
    stdevs[network] = tensor.std(0)

sorted_lists = {}
# Measure conditional variance on position
for network in tqdm(all_networks, desc='cond var'):
    # Sentence lengths:
    lengths = [x.size()[0] for x in all_networks[network]]

    # Turn the sequences into a padded sequence
    sequences = torch.zeros(len(all_networks[network]), max(lengths), 500)
    for i, line in enumerate(all_networks[network]):
        sequences[i, :lengths[i], :] = (line - means[network]) / stdevs[network]

    # Verify that the mean is zero
    concatenated = torch.cat([sequences[i, :lengths[i], :] for i in range(len(sequences))])

    # Create mask tensor, which will encode lengths simultaneously
    length_tensor = torch.Tensor(lengths).view(-1, 1, 1).expand_as(sequences)
    mask = torch.arange(0, sequences.size()[1]).view(1, -1, 1).expand_as(sequences)
    mask = mask.lt(length_tensor).float()

    # samples x seq_len x embedding_size
    # Take position-wise mean across samples
    # This will be (seq_len x embedding_size)
    pos_mean = (sequences.sum(0) / mask.sum(0)).unsqueeze(0).expand_as(sequences)

    # Get conditional variance
    cond_variance = (((sequences - pos_mean) ** 2) * mask).view(-1, 500).sum(0) / sum(lengths)

    # Get a sorted list
    sorted_lists[network] = sorted([
        (i, cond_variance[i]) for i in range(len(cond_variance))
    ], key = lambda x: x[1])

json.dump(sorted_lists, open('results/length-neurons.json', 'w'), indent = 4)
