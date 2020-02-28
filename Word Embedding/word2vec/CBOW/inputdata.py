import re
import numpy as np
# 1.Data Preparation
def load_data(path="../Dataset/VNESEcorpus.txt"):
    with open(path, 'r') as file:
        data = file.read()
    return data

def build_corpus(data, end_sentence='<END>', start_sentence='<START>'):
    corpus = []
    sentences = data.split(".")
    for sentence in sentences:
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        corpus.append(pattern.findall(sentence.lower()))
    return corpus

def build_vocab(corpus):
    vocab = []
    reverse_vocab = {}
    for sentence in corpus:
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
    vocab.sort()

    return vocab


def sample_training(corpus, word2idx, window_size=2):
    training_samples = []
    for sentence in corpus:
        len_sentence = len(sentence)
        for t in range(len_sentence):
            context = []
            for m in range(-window_size, window_size+1, 1):
                if (m != 0) and (t+m >= 0) and (t+m < len_sentence):
                    context.append(word2idx[sentence[t+m]])
            element = [context, word2idx[sentence[t]]]
            training_samples.append(element)
    return training_samples

def to_one_hot(index_of_word, vocab_size):
    tmp = np.zeros(vocab_size)
    tmp[index_of_word] = 1
    return tmp

data = "We are about to study the idea of a computational process.\
    Computational processes are abstract beings that inhabit computers.\
    As they evolve, processes manipulate other abstract things called data.\
    The evolution of a process is directed by a pattern of rules\
    called a program. People create programs to direct processes. In effect,\
    we conjure the spirits of the computer with our spells <3"

tokenized_corpus = build_corpus(data)
vocabulary = build_vocab(tokenized_corpus)
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

# data_sampling = sample_training(tokenized_corpus, word2idx)
# print(data_sampling)