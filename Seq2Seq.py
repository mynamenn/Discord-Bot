from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
data_path = 'data/movie_lines.txt'

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r') as f:
    lines = f.read().split('\n')

alternate = True
maxInputLength = 0
maxTargetLength = 0
for line in lines:
    current_sent = '\t' + line.split('+++$+++')[-1] + '\n'
    if alternate:
        input_texts.append(current_sent)
        if len(current_sent) > maxInputLength:
            maxInputLength = len(current_sent)
        for c in current_sent:
            input_characters.add(c)
        alternate = False
    else:
        target_texts.append(current_sent)
        if len(current_sent) > maxTargetLength:
            maxTargetLength = len(current_sent)
        for c in current_sent:
            target_characters.add(c)
        alternate = True

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = maxInputLength
max_decoder_seq_length = maxTargetLength

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)



