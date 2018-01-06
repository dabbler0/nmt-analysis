import sys
from tqdm import tqdm
from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize.stanford_segmenter import StanfordSegmenter

languages = ['en', 'ar', 'fr', 'es', 'ru']

for language in languages:

    tokenizer = MosesTokenizer()

    f = open('/home/anthony/sls/data/testsets/testset/UNv1.0.testset.%s' % (language,), 'r')
    w = open('/home/anthony/sls/data/testsets/tokenized-test/%s.tok' % (language,), 'w')

    for line in tqdm(f, total=4000, desc=language):
        w.write('%s\n' % (' '.join(tokenizer.tokenize(line),)))

    w.flush()
    w.close()
    f.close()
