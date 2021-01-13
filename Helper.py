import numpy as np


def tanh_element_wise(x):
    """
    Simply returned the hyperbolic tangent of every element in x

    :param x: Numpy array
    :return: Numpy array of elementwise tanh of x
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def encode(word_list, encoding_dict):
    """
    Encodes a list of words, from a string to an integer (which should be uniquely assigned to each unique word)
    This is like one-hot encoding but returns a single index instead of a a series of 0's with a 1 at that index.
    The result of the algorithm will be identical either way.

    :param word_list: A list of strings
    :param encoding_dict: A dictionary, string keys and integer values
    :return: A list of integers.
    """
    index_list = []
    for word in word_list:
        index_list.append(encoding_dict[word])
    return index_list


def embed(index_list, matrix):
    """
    Embeds a given (already one-hot encoded) list of words.

    :param index: An list of integer, (0 <= integer < length of vocab list)
    :param matrix: The embedding matrix, which the algorithm learns
    :return: A list of kx1 numpy arrays, k being the embedding dimension.
    """
    # Get embedding dimension
    embed_dim = matrix.shape[0]

    embedded_vector_list = []
    for index in index_list:
        vector = []
        for i in range(embed_dim):
            vector.append(matrix[i][index])
        vector = np.array(vector).reshape(embed_dim, 1)
        embedded_vector_list.append(vector)

    # Return kx1 numpy array (k is embedding dimension)
    return embedded_vector_list


def cosine_similarity(vec1, vec2):
    return np.matmul(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

