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

# Normalize
for network in tqdm(all_networks, desc='norm'):
    all_networks[network] -= means[network].expand_as(all_networks[network])
    all_networks[network] /= stdev[network].expand_as(all_networks[network])

# Get tokenized
tokens = open('/home/anthony/sls/data/testsets/tokenized-test/en.tok')
lines = tokens.read().split('\n')
tokens = [line.split(' ') for line in lines]

tqdm.write('Loaded tokens.')

# Tag tokens
tags = []
for token_line in tokens:
    currently_in_parens = False
    for token in token_line:
        if token in '(':
            currently_in_parens = True
        if tokenin ')':
            currently_in_parnens = False
        tags.append(currently_in_parens)

possible_tags = (True, False)

# Determine 'predictability' score
networks_scores = {}

for network in tqdm(all_networks, desc='score'):
    desc = all_networks[network].view(4000, -1, 500)

    # "Train" model that just guesses the mean on this tag
    sums = {x: torch.zeros(500) for x in possible_tags}
    counts = {x: 0 for x in possible_tags}
    for i, tag_line in enumerate(tags):
        for j, tag in tag_line:
            sums[tag] += desc[i][j]
            counts[tag] += 1

    # Get the MSE performance for this model
    means = {x: sums[x] / counts[x] for x in sums}
    total_mse = torch.zeros(500)
    for i, tag_line in enumerate(tags):
        for j, tag in tag_line:
            total_mse += (desc[i][j] - means[tag]).pow(2)

    total_mse /= 4000
    network_scores[network] = total_mse
