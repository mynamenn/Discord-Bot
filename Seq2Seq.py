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

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)]
)
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)]
)

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32'
)
decoder_input_data = np.zeros(
    (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32'
)
decoder_target_data = np.zeros(
    (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32'
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t:, input_token_index[char]] = 1
    encoder_input_data[i, t+1:, input_token_index[' ']] = 1
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1
        if t > 0:
            decoder_target_data[i, t-1, target_token_index[char]] = 1
        decoder_target_data[i, t:, target_token_index[' ']] = 1
        decoder_input_data[i, t+1:, target_token_index[' ']] = 1

# Encoder input sequence.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(latent_dim, return_state= True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Set up decoder, using encoder_states as the initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model that maps (encoder_input_data & decoder_input_data) to decoder_target_data.
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Training the model.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('s2s.h5')

# Define sampling models.
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_c, decoder_state_input_h]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Reverse lookup token index.
# We will use [0,0,0,1,0] to look up for a character in decode_sequence.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode seq to obtain initial states.
    states_value = encoder_model.predict(input_seq)

    # Target seq represents the current character.
    target_seq = np.zeros((1, 1, num_decoder_tokens))  #[0,0,0,0,1,0,0,0]
    target_seq[0, 0, target_token_index['\t']] = 1

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # Returns new h and c states.
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1

        # Update decoder states.
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decoded_sentence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)









