from Process import *   # For the dataframes, word lists, and vocab sets of the two bibles.
from Helper import *
import gensim
from gensim.models import word2vec
import numpy as np


# Create word to integer dictionary (isomorphic to one-hot-encoding)
english_vocab_to_int = {word: integer for integer, word in enumerate(english_vocab_set)}
spanish_vocab_to_int = {word: integer for integer, word in enumerate(spanish_vocab_set)}

# For going from embedded space to original word
spanish_int_to_vocab = {integer: word for integer, word in enumerate(spanish_vocab_set)}

# Hyperparameters:
k = 50       # embedding dimension
L = 10        # hidden layer size

# size of vocabulary
s = len(english_vocab_set)
s_ = len(spanish_vocab_set)

# Initializing the network connections
# Encoding matrices
M = np.random.normal(0, 1, (k, s))         # English Embedding matrix

W = np.random.normal(0, 1, (L, k))         # Matrix for new x(t)
U = np.random.normal(0, 1, (L, L))         # Matrix for previous h(t-1)
b = np.random.normal(0, 1, (L, 1))         # Bias for calculating neuron activation
h_0 = np.zeros((L, 1))

# Decoding matrices
M_ = np.random.normal(0, 1, (k, s_))                                   # Spanish embedding matrix
initial_matrix = np.random.normal(0, 1, (L, L))                        # Context -> h_(0)

V_ = np.random.normal(0, 1, (k, L))                                    # Hidden to output matrix
c_ = np.random.normal(0, 1, (k, 1))                                    # Hidden to output bias

U_ = np.random.normal(0, 1, (L, L))                                    # Matrix for previous h_(t-1) to hidden
W_ = np.random.normal(0, 1, (L, k))                                    # Matrix for previous output o_(t-1) to hidden
C_ = np.random.normal(0, 1, (L, L))                                    # Matrix for context vector
b_ = np.random.normal(0, 1, (L, 1))                                    # Bias for hidden node

print("Starting the algorithm: ")

# MAIN ALGORITHM LOOP
for verse_index in range(5):  # range(english_df.shape[0]): # TODO you can change this depending on how much times you
                                                            # have to run the program: the whole range can take hours.
    # Encoder-----------------------------------------------------------------------------------------------------------
    eng_verse_word_list = english_df["Text"][verse_index].split()
    eng_verse_vector_list = embed(encode(eng_verse_word_list, english_vocab_to_int), M)

    # Forward loop
    h = h_0  # initial state (zeros)
    for vec in eng_verse_vector_list:
        A = b + np.matmul(W, vec) + np.matmul(U, h)
        h = tanh_element_wise(A)

    # Our final output from the encoder
    context = h
    print("Context for verse ", verse_index, ": ", context)

    # Decoder-----------------------------------------------------------------------------------------------------------
    esp_verse_word_list = spanish_df["Text"][verse_index].split()
    esp_verse_length = len(esp_verse_word_list)
    esp_verse_vector_list = embed(encode(esp_verse_word_list, spanish_vocab_to_int), M_)

    # Initializing the decoder
    h_ = tanh_element_wise(np.matmul(initial_matrix, context))             # First hidden node value
    y_ = [np.matmul(V_, h_) + c_]                                          # Output list initialized with first value

    # Prediction and loss initialization
    norms = np.linalg.norm(M_ - y_[0], axis=0)
    y_decoded = [np.argmin(norms)]
    loss = [norms[y_decoded[0]]]

    # Looping through the rest of the output verse
    for i, word in enumerate(esp_verse_word_list[1:]):
        # Calculate hidden node and it's output
        h_ = np.matmul(U_, h_) + np.matmul(W_, y_[i-1]) + np.matmul(C_, context) + b_
        y_.append(np.matmul(V_, h_) + c_)

        # Calculate norms between output and all possible embeddings
        norms = np.linalg.norm(M_ - y_[i], axis=0)

        # Add the index of the most similar embedding
        y_decoded.append(np.argmin(norms))

        # Add the loss of the predicted embedding
        loss.append(norms[y_decoded[i]])

    # Print output as words
    print("\nVerse:", english_df["Verse"][verse_index])
    for word_index in y_decoded:
        print(spanish_int_to_vocab[word_index], end=" ")

    # Back propagate (takes to long to run on most processors - big gradient matrices).






