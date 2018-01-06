import torch
import numpy
import json
from tqdm import tqdm
from numpy import newaxis as na
from torch.utils.serialization import load_lua
from itertools import product as p

languages = ['es', 'fr', 'ar', 'ru', 'zh']

all_networks = {}
canonical_ordering = []

for language, version in tqdm(p(languages, [1, 2, 3]), desc='loading', total=len(languages) * 3):
    network_name = 'en-%s-%d' % (language, version)

    canonical_ordering.append(network_name)

    # Load the description of the network
    # This will be a 4000x(sentence_length)x500 matrix.
    # Rehsape to be a (total_tokens)x500 matrix.
    all_networks[network_name] = torch.cat(load_lua(
        '../descriptions/%s.desc.t7' % (network_name,)
    ))

print(canonical_ordering)

# Create enormous data set, which will have
# a second dimension of size 500*15 = 7500
# Transfer it to CUDA.
full_set = torch.cat([all_networks[network] for network in canonical_ordering], dim=1).cuda()

# Whiten the full set
full_set -= full_set.mean(0)
full_set /= full_set.std(0)

# Get covariances
covariances = torch.mm(full_set.t(), full_set) / (full_set.size()[1] - 1)

# Remove full set from memory so that eigenvalues
# can be computed
del full_set
print('Computing eigenvalues now.')

# Get the eigenvalues and eigenvectors
e, v = torch.eig(covariances, eigenvectors = True)

# Save the values
torch.save((e, v), 'eigenvalues_and_vectors.pkl')
