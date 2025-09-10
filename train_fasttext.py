from gensim.models import FastText
from pathlib import Path

#Load data
with open('Iliad_lemmatized.txt', 'r', encoding="utf-8") as f:
    iliad = f.read()
with open('Odyssey_lemmatized.txt', 'r', encoding="utf-8") as f:
    odyssey = f.read()

lemmatized_texts = []
lemmatized_texts.extend(sorted(list(Path('Lemmatized_Data').glob('*'))))
plato_works = {}
for lemmatized_text in lemmatized_texts:
    with open(lemmatized_text, 'r', encoding="utf-8") as f:
        l_t = f.read()
    name = lemmatized_text.name
    plato_works[name] = l_t

#make corpus
def make_sentences_list(text):
    """
    Turns a text in a list of sentences. Note that Ancient Greek uses different punctuation!
    :param text: str
    :return: list of lists
    """
    sentences = []
    sentence = []

    text = text.replace('.', ' .').replace(';', ' ;')
    tokens = text.split()
    for word in tokens:
        if word in [';', '.']:
            if sentence:
                sentences.append(sentence[:-1])
                sentence = []
        else:
            sentence.append(word)
    return sentences

corpus = make_sentences_list(iliad) + make_sentences_list(odyssey)
for name in plato_works.keys():
    corpus.extend(make_sentences_list(plato_works[name]))

#train Fasttext model
fasttext_matrix_path = Path('fasttext_model.bin')

vector_size = 300
window = 4
min_count = 10      # Ignores all words with total frequency lower than this.
workers = 4         # Use 4 CPU cores
sample = 0.00001    # threshold for configuring which higher-frequency words are randomly downsampled.
alpha = 0.025       # Initial learning rate
min_alpha = 0.0001  # minimal value of the learning rate (gensim default)
sg = 1              # Training algorithm: 1 for skip-gram; otherwise CBOW. (gensim default = 0)
hs = 0              # If 1, hierarchical softmax will be used for model training, if 0 not (gensim default)
negative = 5        # number of negative samples (gensim default)
ns_exponent = 0.75  # context distribution smoothing parameter (gensim default)
min_n = 3           # Min subword n-gram length
max_n = 6           # Max subword n-gram length
epochs = 35         # Training iterations (gensim default = 10)

model = FastText(
            sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
            sample=sample, alpha=alpha, min_alpha=min_alpha, sg=sg, hs=hs, negative=negative, ns_exponent=ns_exponent,
            min_n=min_n, max_n=max_n, epochs=epochs)

model.save(str(fasttext_matrix_path))

