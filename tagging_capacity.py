import torch
from torch.utils.serialization import load_lua
from itertools import product as p
from tqdm import tqdm
import json

languages = ['es', 'fr', 'ar', 'ru', 'zh']

networks = {}

for lang, n in tqdm(p(languages, [1, 2, 3]), desc='load', total=len(languages)*3):
    nname = 'en-%s-%d' % (lang, n)
    desc_file = '/home/anthony/sls/descriptions/%s.desc.t7' % (nname,)

    networks[nname] = load_lua(desc_file)

    sample = torch.cat(networks[nname])

    # Normalize description
    mean = sample.mean(0)
    stdev = (sample - mean).pow(2).mean(0).sqrt()

    # In-place normalize
    for x in networks[nname]:
        x.sub_(mean).div_(stdev)

with open('/home/anthony/sls/data/sample/en.tok') as f:
    lines = f.read().split('\n')[:-1]
    lines = [x.split(' ') for x in lines]
    lines = [line for line in lines if len(line) < 250]

def accuracy_score(indices, tag_tensor):
    return indices.eq(tag_tensor.unsqueeze(1).expand_as(indices)).float().mean(0)

def make_f1_scorer(index):
    epsilon = 1e-7

    def f1_score(indices, tag_tensor):
        positives = indices.eq(index).float()
        retrieved = tag_tensor.eq(index).unsqueeze(1).expand_as(indices).float()

        precision = (positives * retrieved).sum(0) / (epsilon + retrieved.sum(0))
        recall = (positives * retrieved).sum(0) / (epsilon + positives.sum(0))

        return 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score

def match_expectation(manual_tag, tags, desc='match', scoring_function=accuracy_score):
    # Tag to index:
    tag2idx = {tag: i for i, tag in enumerate(tags)}

    concatenated_tags = []

    # Sort into buckets
    for i, line in tqdm(enumerate(lines), total=len(lines), desc=desc):
        line_tags = manual_tag(line)
        concatenated_tags.extend([tag2idx[tag] for tag in line_tags])

    tag_tensor = torch.Tensor(concatenated_tags).long().cuda()

    network_accuracies = {}

    for nname in networks:
        # tokens x dim_size
        data = torch.cat(networks[nname]).float().cuda()
        tokens, dim_size = data.size()

        # Get necessary data for mixed Gaussian model
        mean_tensor = torch.stack([
            data.index_select(0, tag_tensor.eq(i).nonzero().squeeze()).mean(0)
            for i in range(len(tags))
        ])

        stdev_tensor = torch.stack([
            data.index_select(0, tag_tensor.eq(i).nonzero().squeeze()).std(0)
            for i in range(len(tags))
        ])

        count_tensor = torch.Tensor([tag_tensor.eq(i).float().mean() for i in range(len(tags))]).cuda()

        count_tensor = torch.log(count_tensor)

        # Do predictions from mixed Gaussian model
        likelihoods = data.unsqueeze(0).expand(len(tags), tokens, dim_size)

        mean_tensor = mean_tensor.unsqueeze(1)#.expand_as(likelihoods)
        stdev_tensor = stdev_tensor.unsqueeze(1)#.expand_as(likelihoods)
        count_tensor = count_tensor.unsqueeze(1).unsqueeze(1)#.expand_as(likelihoods)

        likelihoods = (-(
            (likelihoods - mean_tensor) / stdev_tensor
        ) ** 2) / 2 + count_tensor

        # Indices here should be tokens x dim_size
        maxs, indices = torch.max(likelihoods, dim = 0)

        if nname == 'en-es-1':
            print(indices[:, 232])

        # Accuracies
        accuracies = scoring_function(indices, tag_tensor)
        #indices.eq(tag_tensor.unsqueeze(1).expand(tokens, dim_size)).float().mean(0)

        scores, neurons = torch.sort(accuracies, descending = True)

        scores = scores.cpu().numpy().tolist()
        neurons = neurons.cpu().numpy().tolist()

        network_accuracies[nname] = list(zip(neurons, scores))

    json.dump(
        network_accuracies,
        open('results/attempt-tag-%s.json' % (desc,), 'w'),
        indent=4
    )

# Manual parenthesis tagger
PAREN_OPEN_TOKENS = ('(', '[', '&quot;', '&#91;')
PAREN_CLOSE_TOKENS = (')', ']', '&quot;', '&#93;')

def paren_tagger(tokens):
    tags = []
    currently_in_parens = False
    for token in tokens:
        if (not currently_in_parens) and token in PAREN_OPEN_TOKENS:
            currently_in_parens = True
        elif (currently_in_parens) and token in PAREN_CLOSE_TOKENS:
            currently_in_parens = False
        tags.append(currently_in_parens)
    return tags

import spacy
nlp = spacy.load('en')

# Manual POS tagger
def pos_tagger(tokens):
    doc = nlp.tokenizer.tokens_from_list(tokens)
    nlp.tagger(doc)
    return [token.pos_ for token in doc]

# Manual noun-phrase position tagger
def noun_tagger(tokens):
    doc = nlp.tokenizer.tokens_from_list(tokens)
    nlp.tagger(doc)
    nlp.parser(doc)

    # Mark IOB for noun chunks
    tags = ['O' for token in tokens]
    for chunk in doc.noun_chunks:
        tags[chunk.start] = 'B'
        for j in range(chunk.start + 1, chunk.end):
            tags[j] = 'I'

    return tags

from spacy.parts_of_speech import VERB
# Manual negation tagger
def negation_tagger(tokens):
    doc = nlp.tokenizer.tokens_from_list(tokens)
    nlp.tagger(doc)
    nlp.parser(doc)

    def is_negated(token):
        while token.dep_ != 'ROOT':
            if any(child.dep_ == 'neg' for child in token.children):
                return True
            token = token.head
        return False

    return [is_negated(token) for token in doc]

# Parenthesis matching
match_expectation(
    paren_tagger,
    (True, False),
    desc = 'parens',
    scoring_function = make_f1_scorer(0)
)

# POS matching
'''
match_expectation(
    pos_tagger,
    (
        'PUNCT',
        'SYM',
        'X',
        'ADJ',
        'VERB',
        'CONJ',
        'CCONJ',
        'SCONJ',
        'NUM',
        'DET',
        'ADV',
        'ADP',
        'ADJ',
        'NOUN',
        'PROPN',
        'PART',
        'PRON',
        'SPACE',
        'PART',
        'INTJ'
    ),
    desc = 'pos'
)
'''

# Noun phrase position matching
match_expectation(
    noun_tagger,
    ('I', 'O', 'B'),
    desc = 'noun'
    #scoring_function = make_f1_scorer(0)
)

# Negation matching
'''
match_expectation(
    negation_tagger,
    (True, False),
    desc = 'neg'
)
'''
