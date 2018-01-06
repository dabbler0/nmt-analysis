import os
import subprocess
import torch
import h5py
import json
import nltk

lang = 'es'
version = 1
model_name = 'en-%s-2m-%d' % (lang, version)
network = 'en-%s-%d' % (lang, version)
projection_file = '/home/anthony/sls/src/tmp.json'

with open('results/annotated-clusters-min.json', 'r') as f:
    clusters = json.load(f)

network_orders = {network: [x[0] for x in clusters[network]] for network in clusters}

# BEGIN TEST
# =========
order = network_orders[network]

top_bleu_scores = {}
bottom_bleu_scores = {}

def test(projection_matrix, label):
    with open(projection_file, 'w') as pf:
        json.dump(projection_matrix, pf)

    tmp_file_name = '/home/anthony/sls/src/tmp/%s-%s-output.txt' % (model_name, label)

    language = model_name[3:5]
    true_file_name = '/home/anthony/sls/data/testsets/tokenized-test/%s.tok' % language

    # Run the tests
    subprocess.call(
        [   '/home/anthony/torch/install/bin/th',
            '/home/anthony/sls/seq2seq-attn/evaluate.lua',
            '-model', '/home/anthony/sls/models/%s-model_final.t7' % model_name,
            '-src_file', '/home/anthony/sls/data/testsets/tokenized-test/en.tok',
            '-output_file', tmp_file_name,
            '-projection', projection_file,
            '-src_dict', '/home/anthony/sls/dicts/%s.src.dict' % model_name,
            '-targ_dict', '/home/anthony/sls/dicts/%s.targ.dict' % model_name,
            '-replace_unk', '1',
            '-gpuid', '1'
        ],
        cwd = '/home/anthony/sls/seq2seq-attn/'
    )

    # Compute a BLEU score of the results.
    average_bleu_score = 0
    samples = 0
    with open(tmp_file_name) as tmp_file:
        with open(true_file_name) as true_file:
            for tmp_line, true_line in zip(tmp_file, true_file):
                tmp_line = tmp_line.split(' ')
                true_line = true_line.split(' ')

                bleu_score = nltk.translate.bleu_score.sentence_bleu([true_line], tmp_line)

                print(bleu_score)

                average_bleu_score += bleu_score
                samples += 1

    average_bleu_score /= samples
    return average_bleu_score


for threshold in [50 * x for x in range(1, 11)]:
    # Create the projection matrix and pass it
    # to the decoder

    # Starts as identity matrix
    projection_matrix = [[(1 if x == y else 0) for x in range(500)] for y in range(500)]

    # Zero out killed dimensions
    for i in range(min(threshold, 500)):
        projection_matrix[order[i]][order[i]] = 0

    top_bleu_scores[threshold] = test(projection_matrix, 'top-%d' % threshold)

    # Starts as identity matrix
    projection_matrix = [[(1 if x == y else 0) for x in range(500)] for y in range(500)]

    # Zero out killed dimensions
    for i in range(1, min(threshold, 500) + 1):
        projection_matrix[order[-i]][order[-i]] = 0

    bottom_bleu_scores[threshold] = test(projection_matrix, 'bottom-%d' % threshold)

# END TEST
# ========

json.dump((top_bleu_scores, bottom_bleu_scores), 'results/bleu-score-%s.json' % network)
