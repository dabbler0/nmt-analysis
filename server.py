import torch
import numpy
import json
from tqdm import tqdm
from torch.utils.serialization import load_lua
from itertools import product as p

'''
LOAD NETWORKS
'''

languages = ['es', 'fr', 'ar', 'ru', 'zh']
versions = [1, 2, 3]

include_pca = False #True #False

all_networks = {}

canonical_ordering = []

for language, version in tqdm(
            p(languages, versions),
            desc = 'loading',
            total = len(languages) * len(versions)):

    # Standard network name
    network_name = 'en-%s-%d' % (language, version)

    canonical_ordering.append(network_name)

    # Load as 4000x(sentence_length)x500 matrix
    all_networks[network_name] = load_lua('../descriptions/%s.desc.t7' % network_name)

means = {}
variances = {}

# Eigenvectors and values
e, v = torch.load('eigenvalues_and_vectors.pkl')

# transforms
cca_transforms = torch.load('svcca-99.pkl')

# Sort
e, indices = torch.abs(e[:, 0]).sort(descending = True)
v = v[:, indices]

pca_list = []

# Get means and variances
for network in tqdm(all_networks, desc = 'norm, pca'):
    # large number x 500
    concatenated = torch.cat(all_networks[network], dim = 0).cuda()
    means[network] = concatenated.mean(0)
    variances[network] = concatenated.std(0)

    # PCA:
    if include_pca:
        pca_concatenated = (concatenated - means[network] / variances[network]).cpu()

        pca_list.append(pca_concatenated)

    means[network] = means[network].cpu()
    variances[network] = variances[network].cpu()

if include_pca:
    pca_network = torch.cat(pca_list, dim=1)

    # Perform projection in several (20 right now) chunks
    N = 5
    chunk_size = pca_network.size()[0] // N + 1
    transformed_chunks = []
    for i in tqdm(range(N), total=N, desc='transform'):
        transformed_chunks.append(
            torch.mm(
                pca_network[i*chunk_size:(i + 1)*chunk_size, :].cuda(), v
            ).cpu()
        )

    print('Concatenating')
    pca_network = torch.cat(transformed_chunks)

    print('Computing means and variances')
    pca_means = pca_network.mean(0)
    pca_variances = pca_network.std(0)

    total = 0
    pca_list = []
    for line in tqdm(all_networks['en-es-1'], 'deconcat'):
        pca_list.append(pca_network[total:total + len(line)])
        total += len(line)

    pca_network = pca_list

    print('Done.')

# X X^T = V E V^T
# V^T X X^T V = E
# Therefore V^T X is orthogonal

'''
SERVER CLASS
'''

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import os

class VisualizationServer(BaseHTTPRequestHandler):

    STATIC_CONTENT_PATH = {
        '': 'html/index.html',
        '/': 'html/index.html',
        '/lines.txt': '../data/testsets/tokenized-test/en.tok'
    }

    MIME_TYPES = {
        '.html': 'text/html',
        '.tok': 'text/plain'
    }

    def do_GET(self):
        # 200 OK
        self.send_response(200)

        # Parse path
        path = urlparse(self.path)
        query = parse_qs(path.query)

        # Determine if path leads to static content.
        if path.path in VisualizationServer.STATIC_CONTENT_PATH:
            _, ext = os.path.splitext(VisualizationServer.STATIC_CONTENT_PATH[path.path])

            self.send_header('Content-type', VisualizationServer.MIME_TYPES[ext])
            self.end_headers()

            with open(VisualizationServer.STATIC_CONTENT_PATH[path.path]) as f:
                self.wfile.write(f.read().encode('ascii', 'xmlcharrefreplace'))

        # Otherwise, run one of our endpoints.
        # We can request visualizations for individual neurons.
        if path.path == '/activations':
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # Get standard identifier.
            identifier = query['neuron'][0]

            # Identify the wanted neuron
            network, neuron = identifier.split(':')

            neuron = int(neuron)

            if network == 'pca' and include_pca:
                # Get the desired neuron activations
                activations = [x[:, neuron].numpy().tolist() for x in pca_network]
                mean = pca_means[neuron]
                variance = pca_variances[neuron]

                # Write
                self.wfile.write(json.dumps({
                    'activations': activations,
                    'mean': mean,
                    'variance': variance
                }).encode('ascii'))
            elif '/' in network:
                a, b = network.split('/')

                pair = (a, b) if (a, b) in cca_transforms else (b, a)

                transform = cca_transforms[pair][a].cpu()

                # Perform the transform on the fly
                activations = [
                    torch.mm(x, transform)[:, neuron] for x in all_networks[a]
                ]

                # Get mean and variance on the fly
                concat = torch.cat(activations)
                mean = concat.mean()
                variance = concat.std()

                activations = [x.numpy().tolist() for x in activations]

                self.wfile.write(json.dumps({
                    'activations': activations,
                    'mean': mean,
                    'variance': variance
                }).encode('ascii'))

            else:
                # Get the desired neuron activations
                activations = [x[:, neuron].numpy().tolist() for x in all_networks[network]]
                mean = means[network][neuron]
                variance = variances[network][neuron]

                # Write
                self.wfile.write(json.dumps({
                    'activations': activations,
                    'mean': mean,
                    'variance': variance
                }).encode('ascii'))

print('Running server on 8080.')
server_address = ('', 8080)
httpd = HTTPServer(server_address, VisualizationServer)
httpd.serve_forever()

