"""
Class for encoding variable-length peptides to fixed-size numerical matrices
"""
from __future__ import (
    print_function,
    division,
    absolute_import,
)

import math
from six import string_types
from functools import partial

import numpy
import pandas

from amino_acid import *


class EncodingError(ValueError):
    """
    Exception raised when peptides cannot be encoded
    """
    def __init__(self, message, supported_peptide_lengths):
        self.supported_peptide_lengths = supported_peptide_lengths
        ValueError.__init__(
            self,
            message + " Supported lengths: %s - %s." % supported_peptide_lengths)


class EncodableSequences(object):
    """
    Class for encoding variable-length peptides to fixed-size numerical matrices
    
    This class caches various encodings of a list of sequences.

    In practice this is used only for peptides. To encode MHC allele sequences,
    see AlleleEncoding.
    """
    unknown_character = "X"

    @classmethod
    def create(klass, sequences):
        """
        Factory that returns an EncodableSequences given a list of
        strings. As a convenience, you can also pass it an EncodableSequences
        instance, in which case the object is returned unchanged.
        """
        if isinstance(sequences, klass):
            return sequences
        return klass(sequences)

    def __init__(self, sequences):
        if not all(isinstance(obj, string_types) for obj in sequences):
            raise ValueError("Sequence of strings is required")
        self.sequences = numpy.array(sequences)
        lengths = pandas.Series(self.sequences, dtype=numpy.object_).str.len()

        self.min_length = lengths.min()
        self.max_length = lengths.max()

        self.encoding_cache = {}
        self.fixed_sequence_length = None
        if len(self.sequences) > 0 and all(
                len(s) == len(self.sequences[0]) for s in self.sequences):
            self.fixed_sequence_length = len(self.sequences[0])

    def __len__(self):
        return len(self.sequences)

    def variable_length_to_fixed_length_categorical(
            self,
            alignment_method="processing",
            left_edge=4,
            right_edge=4,
            max_length=15):
        """
        Encode variable-length sequences to a fixed-size index-encoded (integer)
        matrix.

        See `sequences_to_fixed_length_index_encoded_array` for details.
        
        Parameters
        ----------
        alignment_method : string
            One of "pad_middle" or "left_pad_right_pad"
        left_edge : int, size of fixed-position left side
            Only relevant for pad_middle alignment method
        right_edge : int, size of the fixed-position right side
            Only relevant for pad_middle alignment method
        max_length : maximum supported peptide length

        Returns
        -------
        numpy.array of integers with shape (num sequences, encoded length)

        For pad_middle, the encoded length is max_length. For left_pad_right_pad,
        it's 3 * max_length.
        """

        cache_key = (
            "fixed_length_categorical",
            alignment_method,
            left_edge,
            right_edge,
            max_length)

        if cache_key not in self.encoding_cache:
            fixed_length_sequences = (
                self.sequences_to_fixed_length_index_encoded_array(
                    self.sequences,
                    alignment_method=alignment_method,
                    left_edge=left_edge,
                    right_edge=right_edge,
                    max_length=max_length))
            self.encoding_cache[cache_key] = fixed_length_sequences
        return self.encoding_cache[cache_key]

    def variable_length_to_fixed_length_vector_encoding(
            self,
            vector_encoding_name,
            alignment_method="processing",
            left_edge=4,
            right_edge=4,
            max_length=15,
            trim=False,
            pssm=False,
            allow_unsupported_amino_acids=True):
        """
        Encode variable-length sequences to a fixed-size matrix. Amino acids
        are encoded as specified by the vector_encoding_name argument.

        See `sequences_to_fixed_length_index_encoded_array` for details.

        See also: variable_length_to_fixed_length_categorical.

        Parameters
        ----------
        vector_encoding_name : string
            How to represent amino acids.
            One of "BLOSUM62", "one-hot", etc. Full list of supported vector
            encodings is given by available_vector_encodings().
        alignment_method : string
            One of "pad_middle" or "left_pad_right_pad"
        left_edge : int
            Size of fixed-position left side.
            Only relevant for pad_middle alignment method
        right_edge : int
            Size of the fixed-position right side.
            Only relevant for pad_middle alignment method
        max_length : int
            Maximum supported peptide length
        trim : bool
            If True, longer sequences will be trimmed to fit the maximum
            supported length. Not supported for all alignment methods.
        allow_unsupported_amino_acids : bool
            If True, non-canonical amino acids will be replaced with the X
            character before encoding.

        Returns
        -------
        numpy.array with shape (num sequences, encoded length, m)

        where
            - m is the vector encoding length (usually 21).
            - encoded length is max_length if alignment_method is pad_middle;
              3 * max_length if it's left_pad_right_pad.
        """
        cache_key = (
            "fixed_length_vector_encoding",
            vector_encoding_name,
            alignment_method,
            left_edge,
            right_edge,
            max_length,
            trim,
            pssm,
            allow_unsupported_amino_acids)
        if cache_key not in self.encoding_cache:
            fixed_length_sequences = (
                self.sequences_to_fixed_length_index_encoded_array(
                    self.sequences,
                    alignment_method=alignment_method,
                    left_edge=left_edge,
                    right_edge=right_edge,
                    max_length=max_length,
                    trim=trim,
                    allow_unsupported_amino_acids=allow_unsupported_amino_acids))
#            result = fixed_vectors_encoding(fixed_length_sequences, 
#                                            ENCODING_DATA_FRAMES[vector_encoding_name],
#                                            pssm=pssm)
            assert fixed_length_sequences.shape[0] == len(self.sequences)
            self.encoding_cache[cache_key] = fixed_length_sequences #result
        return self.encoding_cache[cache_key]

    @classmethod
    def sequences_to_fixed_length_index_encoded_array(
            klass,
            sequences,
            alignment_method="processing",
            left_edge=4,
            right_edge=4,
            max_length=15,
            trim=False,
            allow_unsupported_amino_acids=False):
        """
        Encode variable-length sequences to a fixed-size index-encoded (integer)
        matrix.

        Parameters
        ----------
        sequences : list of string
        alignment_method : string
            One of "pad_middle", "left_pad_right_pad", "processing"
        left_edge : int
            Size of fixed-position left side.
            Only relevant for pad_middle alignment method
        right_edge : int
            Size of the fixed-position right side.
            Only relevant for pad_middle alignment method
        max_length : int
            maximum supported peptide length
        trim : bool
            If True, longer sequences will be trimmed to fit the maximum
            supported length. Not supported for all alignment methods.
        allow_unsupported_amino_acids : bool
            If True, non-canonical amino acids will be replaced with the X
            character before encoding.

        Returns
        -------
        numpy.array of integers with shape (num sequences, encoded length)

        For pad_middle, the encoded length is max_length. For left_pad_right_pad,
        it's 2 * max_length. For left_pad_centered_right_pad, it's
        3 * max_length.
        """
        if allow_unsupported_amino_acids:  # if unrecog AA provided (eg J) replace with X
            fill_value = AMINO_ACID_INDEX['X']

            def get_amino_acid_index(a):
                return AMINO_ACID_INDEX.get(a, fill_value)
        else:
            get_amino_acid_index = AMINO_ACID_INDEX.__getitem__

        result = None
        if alignment_method == 'processing':
            #  create blank array where rows are the supplied sequences and 
            #  column length is equal to the desired length
            result = numpy.full(
                fill_value=AMINO_ACID_INDEX['X'],
                shape=(len(sequences), max_length),
                dtype="int32")
            
            df = pandas.DataFrame({"peptide": sequences}, dtype=numpy.object_)
            df["length"] = df.peptide.str.len()
                
            for (length, sub_df) in df.groupby("length"): 
                diff = max_length - length
                #  identify peptides which are shorter than the max length to pad center
                if diff > 0:
                    split = length // 2
                    if length % 2 != 0:  # if length of peptide is odd
                        split += 1  # make larger end the n terminal end
                    #  create pad that puts 'X' at the middle of the max len peptide,
                    #   where pad is equal to the diff btwn max len and actual len
                    xpad = 'X' * diff
                    sub_df["peptide"] = sub_df["peptide"].str[:split] + xpad + \
                                        sub_df["peptide"].str[split:]  # put pad in mid
                                        
                #  Identify peptides that are longer than max length to trim center
                elif diff < 0:
                    split1 = split2 = max_length // 2   # calc length that should be kept
                    if max_length % 2 != 0:             # from each end of peptide
                        split1 += 1  # if max_length is odd, keep 1 more from n term side

                    sub_df["peptide"] = sub_df["peptide"].str[:split1] +\
                                        sub_df["peptide"].str[-split2:]  # trim peptides
                
                #  Convert sequence string to array of corresponding amino acid idx vals
                fixed_length_sequences = numpy.stack(
                            sub_df.peptide.map(
                                lambda s: numpy.array([
                                    get_amino_acid_index(char) for char in s
                                ])).values)
                #  map array to result at correct indicies
                result[sub_df.index, :] = fixed_length_sequences
        
        else:
            raise NotImplementedError(
                "Unsupported alignment method: %s" % alignment_method)


        return result
