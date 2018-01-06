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

def match_expectation(manual_tag, tags, desc='match'):
    buckets = {nname: {tag: [] for tag in tags} for nname in networks}

    # Sort into buckets
    for nname, (i, line) in tqdm(
            p(networks, enumerate(lines)), total=len(networks)*len(lines), desc=desc):

        line_tags = manual_tag(line)

        for j, tag in enumerate(line_tags):
            buckets[nname][tag].append(networks[nname][i][j])

    # Get means
    bucket_means = {
        nname: {
            tag: torch.stack(buckets[nname][tag]).mean(0) if len(buckets[nname][tag]) > 0 else torch.zeros(500)
            for tag in tags
        } for nname in tqdm(buckets, desc='means')
    }

    # Get MSEs
    mean_squared_residuals = {
        nname: torch.stack([
            v - bucket_means[nname][tag]
            for tag in buckets[nname]
            for v in buckets[nname][tag]
        ]).pow(2).mean(0)
        for nname in tqdm(buckets, desc='mses')
    }

    annotated_sorted = {
        nname: sorted(
            ((x, mean_squared_residuals[nname][x]) for x in range(500)),
            key = lambda x: x[1]
        ) for nname in tqdm(buckets, desc='annotate')
    }

    json.dump(
        annotated_sorted,
        open('results/manual-correlate-%s.json' % (desc,), 'w'),
        indent=4
    )

# Manual parenthesis tagger
def paren_tagger(tokens):
    tags = []
    currently_in_parens = False
    for token in tokens:
        if token == '(':
            currently_in_parens = True
        elif token == ')':
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
    desc = 'parens'
)

# POS matching
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

# Noun phrase position matching
match_expectation(
    noun_tagger,
    ('I', 'O', 'B'),
    desc = 'noun'
)

match_expectation(
    negation_tagger,
    (True, False),
    desc = 'neg'
)
