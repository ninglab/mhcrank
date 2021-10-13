"""
Class for encoding variable-length flanking and peptides to
fixed-size numerical matrices
"""
from __future__ import (
    print_function, division, absolute_import, )

from six import string_types
from collections import namedtuple
import logging

from encodable_sequences import EncodingError, EncodableSequences

import numpy
import pandas


EncodingResult = namedtuple(
    "EncodingResult", ["array", "cleave_site", "peptide_lengths"])


class FlankingEncoding(object):
    """
    Encode peptides and optionally their N- and C-flanking sequences into fixed
    size numerical matrices. Similar to EncodableSequences but with support
    for flanking sequences and the encoding scheme used by the processing
    predictor.

    Instances of this class have an immutable list of peptides with
    flanking sequences. Encodings are cached in the instances for faster
    performance when the same set of peptides needs to encoded more than once.
    """
    unknown_character = "X"

    def __init__(self, peptides, n_flanks, c_flanks):
        """
        Constructor. Sequences of any lengths can be passed.

        Parameters
        ----------
        peptides : list of strings
            Peptide sequences
        n_flanks : list of strings [same length as peptides]
            Upstream sequences
        c_flanks : list of strings [same length as peptides]
            Downstream sequences
        """
        self.dataframe = pandas.DataFrame({
            "peptide": peptides,
            "n_flank": n_flanks,
            "c_flank": c_flanks,
        }, dtype=str)
        self.encoding_cache = {}

    def __len__(self):
        """
        Number of peptides.
        """
        return self.dataframe.shape[0]
    
    def vector_encode(
            self,
            vector_encoding_name,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            cleave_radius=3,
            allow_unsupported_amino_acids=True,
            throw=True):
        """
        Encode variable-length sequences to a fixed-size matrix.

        Parameters
        ----------
        vector_encoding_name : string
            How to represent amino acids. One of "BLOSUM62", "one-hot", etc.
            See `amino_acid.available_vector_encodings()`.
        peptide_max_length : int
            Maximum supported peptide length.
        n_flank_length : int
            Maximum supported N-flank length
        c_flank_length : int
            Maximum supported C-flank length
        cleave_radius : int
            The number of residues from the c_terminus of the peptide and
            beginning of c_flank that should be extracted for global kernel
            (Default=3)
        allow_unsupported_amino_acids : bool
            If True, non-canonical amino acids will be replaced with the X
            character before encoding.
        throw : bool
            Whether to raise exception on unsupported peptides

        Returns
        -------
        numpy.array with shape (num sequences, length, m)

        where
            - num sequences is number of peptides, i.e. len(self)
            - length is peptide_max_length + n_flank_length + c_flank_length
            - m is the vector encoding length (usually 21).
        """
        cache_key = (
            "vector_encode",
            vector_encoding_name,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            cleave_radius,
            allow_unsupported_amino_acids,
            throw)
        if cache_key not in self.encoding_cache:
            result = self.encode(
                vector_encoding_name=vector_encoding_name,
                df=self.dataframe,
                peptide_max_length=peptide_max_length,
                n_flank_length=n_flank_length,
                c_flank_length=c_flank_length,
                cleave_radius=cleave_radius,
                allow_unsupported_amino_acids=allow_unsupported_amino_acids,
                throw=throw)
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]

    @staticmethod
    def encode(
            vector_encoding_name,
            df,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            cleave_radius,
            allow_unsupported_amino_acids=False,
            throw=True):
        """
        Encode variable-length sequences to a fixed-size matrix.

        Helper function. Users should use `vector_encode`.

        Parameters
        ----------
        vector_encoding_name : string
        df : pandas.DataFrame
        peptide_max_length : int
        n_flank_length : int
        c_flank_length : int
        cleave_radius : int
        allow_unsupported_amino_acids : bool
        throw : bool

        Returns
        -------
        numpy.array
        """        
        
        error_df = df.loc[
            (df.peptide.str.len() > 15) | 
            (df.peptide.str.len() < 8)  # only accomodate pep from 8 to 15 in length
        ]
        if len(error_df) > 0:
            message = (
                "Sequence '%s' (length %d) unsupported. There are %d "
                "total peptides with this length." % (
                    error_df.iloc[0].peptide,
                    len(error_df.iloc[0].peptide),
                    len(error_df)))
            if throw:
                raise EncodingError(
                    message,
                    supported_peptide_lengths=(1, peptide_max_length + 1))
            logging.warning(message)

            # Replace invalid peptides with X's. The encoding will be set to
            # NaNs for these peptides farther below.
            df.loc[error_df.index, "peptide"] = "X" * peptide_max_length

        if n_flank_length > 0:
            n_flanks = df.n_flank.str.pad(
                n_flank_length,
                side="left",
                fillchar="X").str.slice(-n_flank_length).str.upper()
        else:
            n_flanks = pandas.Series([""] * len(df))

        c_flanks = df.c_flank.str.pad(
            c_flank_length,
            side="right",
            fillchar="X").str.slice(0, c_flank_length).str.upper()

            
        peptides = df.peptide.str.upper()
        
        concatenated = n_flanks + peptides + c_flanks
        encoder = EncodableSequences.create(concatenated.values)
        
        array = encoder.variable_length_to_fixed_length_vector_encoding(
            vector_encoding_name=vector_encoding_name,
            alignment_method="processing",
            max_length=n_flank_length + peptide_max_length + c_flank_length,
            allow_unsupported_amino_acids=allow_unsupported_amino_acids)

        array = array.astype("float32")  # So NaNs can be used.
                
        cleave1 = n_flank_length + peptide_max_length - cleave_radius
        cleave2 = -(c_flank_length - cleave_radius)
        cleave_site = array[:, cleave1:cleave2]

        if len(error_df) > 0:
            array[error_df.index] = numpy.nan

        result = EncodingResult(
            array, cleave_site, peptide_lengths=peptides.str.len().values)

        return result
