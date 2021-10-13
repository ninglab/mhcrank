"""
Functions for encoding fixed length sequences of amino acids into various
vector representations, such as one-hot and BLOSUM62.
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import collections
from copy import copy

import pandas as pd
from six import StringIO
import numpy as np
from sklearn.preprocessing import MinMaxScaler


COMMON_AMINO_ACIDS_WITH_UNKNOWN = collections.OrderedDict({
    "X": "Unknown",
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic Acid",
    "C": "Cysteine",
    "E": "Glutamic Acid",
    "Q": "Glutamine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
}.items())

AMINO_ACID_INDEX = dict(
    (letter, i) for (i, letter) in enumerate(COMMON_AMINO_ACIDS_WITH_UNKNOWN))

for (letter, i) in list(AMINO_ACID_INDEX.items()):
    AMINO_ACID_INDEX[letter.lower()] = i  # Support lower-case

AMINO_ACIDS = list(COMMON_AMINO_ACIDS_WITH_UNKNOWN.keys())

BLOSUM62_MATRIX = pd.read_csv(StringIO("""
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  X
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3  0
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  0
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  0
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1  0
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  0
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3  0
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3  0
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1  0
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1  0
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1  0
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2  0
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0 
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3  0
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1  0
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4  0
X  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
"""), sep='\s+').loc[AMINO_ACIDS, AMINO_ACIDS].astype("int8")
assert (BLOSUM62_MATRIX == BLOSUM62_MATRIX.T).all().all()

blos_shape = BLOSUM62_MATRIX.shape
scaler = MinMaxScaler()  # initialize MinMaxScaler
#  flatten so equal values arent assigned different values because
#  rows are treated as distinct features by sklearn
blos_flat = np.asarray(BLOSUM62_MATRIX).flatten()
norm_vals = scaler.fit_transform(blos_flat.reshape(-1,1))  # treat as single feature

#  reproduce blosum matrix as df, now with normalized values
blosum_normalized = pd.DataFrame(norm_vals.reshape(blos_shape),
								 index=AMINO_ACIDS,
								 columns=AMINO_ACIDS)


#  Data obtained from AAIndex -- following preprocessing normalization (minmaxscaler)
#  Amino Acid : [hydrophocity, polorizability, isoelectric point, volume, 
#                molecular weight, sterics, helix probability, sheet probability]
#  X represents an unknown amino acid and its properties are defined
#  as the average of the properties of the 20 common amino acids
pc = {
    'A': [0.230, 0.112, 0.404, 0.164, 0.108, 0.511, 0.875, 0.248],
    'C': [0.404, 0.313, 0.285, 0.323, 0.357, 0.608, 0.312, 0.566],
    'D': [0.174, 0.257, 0.0, 0.324, 0.449, 0.745, 0.542, 0.159],
    'E': [0.177, 0.369, 0.056, 0.488, 0.558, 0.667, 1.0, 0.0],
    'F': [0.762, 0.709, 0.339, 0.783, 0.698, 0.686, 0.552, 0.593],
    'G': [0.026, 0.0, 0.400, 0.0, 0.0, 0.0, 0.114, 0.310],
    'H': [0.230, 0.562, 0.603, 0.561, 0.620, 0.686, 0.614, 0.345],
    'I': [0.838, 0.455, 0.407, 0.663, 0.434, 1.0, 0.583, 0.841],
    'K': [0.434, 0.535, 0.872, 0.694, 0.550, 0.667, 0.729, 0.212],
    'L': [0.577, 0.455, 0.402, 0.663, 0.434, 0.961, 0.719, 0.628],
    'M': [0.445, 0.540, 0.372, 0.620, 0.574, 0.765, 0.969, 0.460],
    'N': [0.023, 0.328, 0.330, 0.398, 0.442, 0.745, 0.385, 0.080],
    'P': [0.736, 0.320, 0.442, 0.376, 0.310, 0.353, 0.0, 0.071],
    'Q': [0.0, 0.440, 0.360, 0.539, 0.232, 0.667, 0.646, 0.398],
    'R': [0.226, 0.712, 1.0, 0.735, 0.767, 0.667, 0.5, 0.283],
    'S': [0.019, 0.152, 0.364, 0.188, 0.232, 0.520, 0.229, 0.345],
    'T': [0.019, 0.264, 0.362, 0.352, 0.341, 0.490, 0.302, 0.575],
    'V': [0.498, 0.342, 0.399, 0.492, 0.326, 0.745, 0.437, 1.0],
    'W': [1.0, 1.0, 0.390, 1.0, 1.000, 0.686, 0.469, 0.575],
    'Y': [0.709, 0.729, 0.362, 0.806, 0.822, 0.686, 0.281, 0.619],
    'X': [0.374, 0.430, 0.408, 0.508, 0.513, 0.647, 0.510, 0.416]
}

PC = pd.DataFrame(pc)

ENCODING_DATA_FRAMES = {
    "BLOSUM62": BLOSUM62_MATRIX,
    "blosum_norm": blosum_normalized,
    "PC": PC,  # encode physiochemical properties for amino acids
    "PC_blosum": PC.append(blosum_normalized, ignore_index=True)  # encode blosum PC combo
}


def available_vector_encodings():
    """
    Return list of supported amino acid vector encodings.

    Returns
    -------
    list of string

    """
    return list(ENCODING_DATA_FRAMES)


def vector_encoding_length(name):
    """
    Return the length of the given vector encoding.

    Parameters
    ----------
    name : string

    Returns
    -------
    int
    """
    return ENCODING_DATA_FRAMES[name].shape[1]


def index_encoding(sequences, letter_to_index_dict):
    """
    Encode a sequence of same-length strings to a matrix of integers of the
    same shape. The map from characters to integers is given by
    `letter_to_index_dict`.

    Given a sequence of `n` strings all of length `k`, return a `k * n` array where
    the (`i`, `j`)th element is `letter_to_index_dict[sequence[i][j]]`.

    Parameters
    ----------
    sequences : list of length n of strings of length k
    letter_to_index_dict : dict : string -> int

    Returns
    -------
    numpy.array of integers with shape (`k`, `n`)
    """
    df = pd.DataFrame(iter(s) for s in sequences)
    result = df.replace(letter_to_index_dict)
    return result.values


def fixed_vectors_encoding(index_encoded_sequences, letter_to_vector_df):
    """
    Given a `n` x `k` matrix of integers such as that returned by `index_encoding()` and
    a dataframe mapping each index to an arbitrary vector, return a `n * k * m`
    array where the (`i`, `j`)'th element is `letter_to_vector_df.iloc[sequence[i][j]]`.

    The dataframe index and columns names are ignored here; the indexing is done
    entirely by integer position in the dataframe.

    Parameters
    ----------
    index_encoded_sequences : `n` x `k` array of integers
    letter_to_vector_df : pandas.DataFrame of shape (`alphabet size`, `m`)

    Returns
    -------
    numpy.array of integers with shape (`n`, `k`, `m`)
    """
    (num_sequences, sequence_length) = index_encoded_sequences.shape
    n_features = letter_to_vector_df.shape[0]
    encode_residues = lambda seq, encode: np.array([encode.iloc[:,i] for i in seq])
    residues = index_encoded_sequences.reshape((-1,))
    result = encode_residues(residues, letter_to_vector_df)
    zeros=None
    target_shape = (num_sequences, sequence_length, n_features)
    return result.reshape(target_shape)

