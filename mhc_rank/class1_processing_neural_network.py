"""
Antigen processing neural network implementation
"""

from __future__ import print_function
import collections, numpy, os, time

from hyperparameters import HyperparameterDefaults
from amino_acid import *
from flanking_encoding import FlankingEncoding
from common import configure_tensorflow

DEFAULT_PREDICT_BATCH_SIZE = 4096
if os.environ.get("MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"):
    DEFAULT_PREDICT_BATCH_SIZE = int(os.environ[
        "MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"
    ])
    logging.info(
        "Configured default predict batch size: %d" % DEFAULT_PREDICT_BATCH_SIZE)

class Class1ProcessingNeuralNetwork(object):
    """
    A neural network for antigen processing prediction
    """
    network_hyperparameter_defaults = HyperparameterDefaults(
        amino_acid_encoding="BLOSUM62",
        learn_embedding='no',
        learned_embed_dims=10,
        peptide_max_length=15,
        n_flank_length=5,
        c_flank_length=5,
        cleave_radius=3,
        flanking_averages=True,
        convolutional_filters=16,
        convolutional_kernel_size=8,
        convolutional_activation="tanh",
        convolutional_kernel_l1_l2=[0.0001, 0.0001],
        dropout_rate=0.5,
        post_convolutional_dense_layer_sizes=[],
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=500,
        validation_split=0.1,
        early_stopping=True,
        minibatch_size=256,
    )
    """
    Hyperparameters for neural network training.
    """

    early_stopping_hyperparameter_defaults = HyperparameterDefaults(
        patience=30,
        min_delta=0.0,
    )
    """
    Hyperparameters for early stopping.
    """

    compile_hyperparameter_defaults = HyperparameterDefaults(
        optimizer="adam",
        loss_function="binary_crossentropy",
        learning_rate=None,
    )
    """
    Loss and optimizer hyperparameters. Any values supported by keras may be
    used.
    """

    auxiliary_input_hyperparameter_defaults = HyperparameterDefaults(
    )
    """
    Allele feature hyperparameters.
    """

    hyperparameter_defaults = network_hyperparameter_defaults.extend(
        fit_hyperparameter_defaults).extend(
        early_stopping_hyperparameter_defaults).extend(
        compile_hyperparameter_defaults).extend(
        auxiliary_input_hyperparameter_defaults)

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters)
        self._network = None
        self.network_json = None
        self.network_weights = None
        self.fit_info = []

    @property
    def sequence_lengths(self):
        """
        Supported maximum sequence lengths

        Returns
        -------
        dict of string -> int

        Keys are "peptide", "n_flank", "c_flank". Values give the maximum
        supported sequence length.
        """
        return {
            "peptide": self.hyperparameters['peptide_max_length'],
            "n_flank": self.hyperparameters['n_flank_length'],
            "c_flank": self.hyperparameters['c_flank_length'],
        }

    def network(self):
        """
        Return the keras model associated with this network.
        """
        if self._network is None and self.network_json is not None:
            # NOTE
            # Instead of calling:
            #   from tensorflow.keras.models import model_from_json
            #   self._network = model_from_json(self.network_json)
            # We are re-creating the network here using the hyperparameters.
            # This is because the network uses Lambda layers, which break
            # when serialized between python versions. The disadvantage is
            # that we can more easily lose backward compatability.
            self._network = self.make_network(
                **self.network_hyperparameter_defaults.subselect(
                    self.hyperparameters))

            if self.network_weights is not None:
                self._network.set_weights(self.network_weights)
        return self._network

    def update_network_description(self):
        """
        Update self.network_json and self.network_weights properties based on
        this instances's neural network.
        """
        if self._network is not None:
            self.network_json = self._network.to_json()
            self.network_weights = self._network.get_weights()

    def fit(
            self,
            sequences,
            targets,
            sample_weights=None,
            shuffle_permutation=None,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0):
        """
        Fit the neural network.

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and upstream/downstream flanking sequences
        targets : list of float
            1 indicates hit, 0 indicates decoy
        sample_weights : list of float
            If not specified all samples have equal weight.
        shuffle_permutation : list of int
            Permutation (integer list) of same length as peptides and affinities
            If None, then a random permutation will be generated.
        verbose : int
            Keras verbosity level
        progress_callback : function
            No-argument function to call after each epoch.
        progress_preamble : string
            Optional string of information to include in each progress update
        progress_print_interval : float
            How often (in seconds) to print progress update. Set to None to
            disable.
        """
        x_dict = self.network_input(sequences)

        # Shuffle
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(targets))
        targets = numpy.array(targets)[shuffle_permutation]
        assert numpy.isnan(targets).sum() == 0, targets
        if sample_weights is not None:
            sample_weights = numpy.array(sample_weights)[shuffle_permutation]
        for key in list(x_dict):
            x_dict[key] = x_dict[key][shuffle_permutation]

        fit_info = collections.defaultdict(list)

        if self._network is None:
            print("Making empty net")
            self._network = self.make_network(
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters))
            if verbose > -1:
                self._network.summary()

        self.network().compile(
            loss=self.hyperparameters['loss_function'],
            optimizer=self.hyperparameters['optimizer'])

        last_progress_print = None
        min_val_loss_iteration = None
        min_val_loss = None
        start = time.time()
        for i in range(self.hyperparameters['max_epochs']):
            epoch_start = time.time()
            fit_history = self.network().fit(
                x_dict,
                targets,
                validation_split=self.hyperparameters['validation_split'],
                batch_size=self.hyperparameters['minibatch_size'],
                epochs=i + 1,
                sample_weight=sample_weights,
                initial_epoch=i,
                verbose=verbose)
            epoch_time = time.time() - epoch_start

            for (key, value) in fit_history.history.items():
                fit_info[key].extend(value)

            # Print progress no more often than once every few seconds.
            if progress_print_interval is not None and (
                    not last_progress_print or (
                        time.time() - last_progress_print
                        > progress_print_interval)):
                print((progress_preamble + " " +
                       "Epoch %3d / %3d [%0.2f sec]: loss=%g. "
                       "Min val loss (%s) at epoch %s" % (
                           i,
                           self.hyperparameters['max_epochs'],
                           epoch_time,
                           fit_info['loss'][-1],
                           str(min_val_loss),
                           min_val_loss_iteration)).strip())
                last_progress_print = time.time()

            if self.hyperparameters['validation_split']:
                val_loss = fit_info['val_loss'][-1]

                if min_val_loss is None or (
                        val_loss < min_val_loss - self.hyperparameters['min_delta']):
                    min_val_loss = val_loss
                    min_val_loss_iteration = i

                if self.hyperparameters['early_stopping']:
                    threshold = (
                        min_val_loss_iteration +
                        self.hyperparameters['patience'])
                    if i > threshold:
                        if progress_print_interval is not None:
                            print((progress_preamble + " " +
                                "Stopping at epoch %3d / %3d: loss=%g. "
                                "Min val loss (%g) at epoch %s" % (
                                    i,
                                    self.hyperparameters['max_epochs'],
                                    fit_info['loss'][-1],
                                    (
                                        min_val_loss if min_val_loss is not None
                                        else numpy.nan),
                                    min_val_loss_iteration)).strip())
                        break

            if progress_callback:
                progress_callback()

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = len(sequences.dataframe)
        self.fit_info.append(dict(fit_info))

        if verbose > -1:
            print(
                "Output weights",
                *numpy.array(
                    self.network().get_layer(
                        "output_final").get_weights()).flatten())

    def predict(
            self,
            peptides,
            n_flanks=None,
            c_flanks=None,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        """
        Predict antigen processing.

        Parameters
        ----------
        peptides : list of string
            Peptide sequences
        n_flanks : list of string
            Upstream sequence before each peptide
        c_flanks : list of string
            Downstream sequence after each peptide
        batch_size : int
            Prediction keras batch size.

        Returns
        -------
        numpy.array

        Processing scores. Range is 0-1, higher indicates more favorable
        processing.
        """
        if n_flanks is None:
            n_flanks = [""] * len(peptides)
        if c_flanks is None:
            c_flanks = [""] * len(peptides)

        sequences = FlankingEncoding(
            peptides=peptides, n_flanks=n_flanks, c_flanks=c_flanks)
        return self.predict_encoded(sequences=sequences, batch_size=batch_size)

    def predict_encoded(
            self,
            sequences,
            throw=True,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        """
        Predict antigen processing.

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and flanking sequences
        throw : boolean
            Whether to throw exception on unsupported peptides
        batch_size : int
            Prediction keras batch size.

        Returns
        -------
        numpy.array
        """
        x_dict = self.network_input(sequences, throw=throw)
        raw_predictions = self.network().predict(
            x_dict, batch_size=batch_size)
        predictions = numpy.squeeze(raw_predictions).astype("float64")
        return predictions

    def network_input(self, sequences, throw=True):
        """
        Encode peptides to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Parameters
        ----------
        sequences : FlankingEncoding
            Peptides and flanking sequences
        throw : boolean
            Whether to throw exception on unsupported peptides

        Returns
        -------
        numpy.array
        """
        encoded = sequences.vector_encode(
            vector_encoding_name=self.hyperparameters['amino_acid_encoding'],
            peptide_max_length=self.hyperparameters['peptide_max_length'],
            n_flank_length=self.hyperparameters['n_flank_length'],
            c_flank_length=self.hyperparameters['c_flank_length'],
            cleave_radius=self.hyperparameters['cleave_radius'],
            allow_unsupported_amino_acids=True,
            throw=throw)
            
        result = {
            "sequence": encoded.array,
            "cleave_site": encoded.cleave_site,
            "peptide_length": encoded.peptide_lengths,
        }
        return result

    def make_network(
            self,
            amino_acid_encoding,
            learn_embedding,
            learned_embed_dims,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            cleave_radius,
            flanking_averages,
            convolutional_filters,
            convolutional_kernel_size,
            convolutional_activation,
            convolutional_kernel_l1_l2,
            dropout_rate,
            post_convolutional_dense_layer_sizes):
        """
        Helper function to make a keras network given hyperparameters.
        """

        # We import keras here to avoid tensorflow debug output, etc. unless we
        # are actually about to use Keras.
        configure_tensorflow()
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, Concatenate, Conv1D, Embedding, GlobalMaxPooling1D, Lambda)
        # consider changing lambda to average1dpooling and maxpooling1d
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers, initializers

        model_inputs = {}
        if amino_acid_encoding == 'PC':
            n_features = 8
        elif amino_acid_encoding == 'BLOSUM62' or amino_acid_encoding == 'blosum_norm':
            n_features = 21
        elif amino_acid_encoding == 'PC_blosum':
            n_features = 29
            
        seq_len = peptide_max_length + n_flank_length + c_flank_length
        cleave_len = 2*cleave_radius

        model_inputs['sequence'] = Input(
            shape=(seq_len,),
            dtype='float32',
            name='sequence')
        model_inputs['cleave_site'] = Input(
            shape=(cleave_len),
            dtype='float32',
            name='cleave_site')
        model_inputs['peptide_length'] = Input(
            shape=(1,),
            dtype='float32',
            name='peptide_length')
        
        
        if learn_embedding != 'no':
            sequence_learned = Embedding(input_dim=21,
                                         output_dim=n_features,
                                         mask_zero=True,
                                         input_length=seq_len)(model_inputs['sequence'])
            cleave_learned = Embedding(input_dim=21,
                                       output_dim=n_features,
                                       mask_zero=True,
                                       input_length=cleave_len)(model_inputs['cleave_site'])
            
        if learn_embedding != 'yes':
            def mk_embedding_matrix(encoding, n_features):
                matrix = numpy.zeros((21, n_features))
                encoding = ENCODING_DATA_FRAMES[encoding]
                for i in range(21):
                    matrix[i] = encoding.iloc[:, i]    
                return matrix
            
            embedding_matrix = mk_embedding_matrix(amino_acid_encoding, n_features)
            
            sequence_hc = Embedding(input_dim=21,
                                    output_dim=n_features,
                                    mask_zero=True,
                                    input_length=seq_len,
                                    weights=[embedding_matrix],
                                    trainable=False)(model_inputs['sequence'])
            cleave_hc = Embedding(input_dim=21,
                                  output_dim=n_features,
                                  mask_zero=True,
                                  input_length=cleave_len,
                                  weights=[embedding_matrix],
                                  trainable=False)(model_inputs['cleave_site'])
            
            
        if learn_embedding == 'no':
            seq_embedding = sequence_hc
            celave_embedding = cleave_hc
            
        elif learn_embedding == 'yes':
            seq_embedding = sequence_learned
            cleave_embedding = cleave_learned
            
        elif learn_embedding == 'combo':
            seq_embedding = Concatenate()([sequence_hc, sequence_learned])
            cleave_embedding = Concatenate()([cleave_hc, cleave_learned])
        
        outputs_for_final_dense = []

        # Global kernel, also called CSSK in manuscript.
        global_kernel = GlobalMaxPooling1D(name='global_kernel')(cleave_embedding)
        outputs_for_final_dense.append(global_kernel)

        current_layer = seq_embedding
        current_layer = Conv1D(
            filters=convolutional_filters,
            kernel_size=convolutional_kernel_size,
            kernel_regularizer=regularizers.l1_l2(
                *convolutional_kernel_l1_l2),
            padding="same",
            activation=convolutional_activation,
            name="conv1")(current_layer)
        if dropout_rate > 0:
            current_layer = Dropout(
                name="conv1_dropout",
                rate=dropout_rate,
                noise_shape=(
                    None, 1, int(current_layer.get_shape()[-1])))(
                current_layer)

        convolutional_result = current_layer

        for flank in ["n_flank", "c_flank"]:
            current_layer = convolutional_result
            for (i, size) in enumerate(
                    [post_convolutional_dense_layer_sizes] + [1]):
                current_layer = Conv1D(
                    name="%s_post_%d" % (flank, i),
                    filters=size,
                    kernel_size=1,
                    kernel_regularizer=regularizers.l1_l2(
                        *convolutional_kernel_l1_l2),
                    activation=(
                        "tanh" if size == 1 else convolutional_activation
                    ))(current_layer)
            single_output_result = current_layer

            dense_flank = None
            if flank == "n_flank":
                def cleavage_extractor(x):
                    return x[:, n_flank_length]

                single_output_at_cleavage_position = Lambda(
                    cleavage_extractor, name="%s_cleaved" % flank)(
                    single_output_result)

                def max_pool_over_peptide_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst

                    # We generate a per-sample mask that is 1 for all peptide
                    # positions except the first position, and 0 for all other
                    # positions (i.e. n flank, c flank, and the first peptide
                    # position).
                    starts = n_flank_length + 1
                    limits = n_flank_length + peptide_length
                    row = tf.expand_dims(tf.range(0, x.shape[1]), axis=0)
                    mask = tf.logical_and(
                        tf.greater_equal(row, starts),
                        tf.less(row, limits))

                    # We are assuming that x >= -1. The final activation in the
                    # previous layer should be a function that satisfies this
                    # (e.g. sigmoid, tanh, relu).
                    max_value = tf.reduce_max(
                        (x + 1) * tf.expand_dims(
                            tf.cast(mask, tf.float32), axis=-1),
                        axis=1) - 1

                    # We flip the sign so that initializing the final dense
                    # layer weights to 1s is reasonable.
                    return -1 * max_value

                max_over_peptide = Lambda(
                    max_pool_over_peptide_extractor,
                    name="%s_internal_cleaved" % flank)([
                        single_output_result,
                        peptide_max_length
                    ])

                def flanking_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst

                    # mask is 1 for n_flank positions and 0 elsewhere.
                    starts = 0
                    limits = n_flank_length
                    row = tf.expand_dims(tf.range(0, x.shape[1]), axis=0)
                    mask = tf.logical_and(
                        tf.greater_equal(row, starts),
                        tf.less(row, limits))

                    # We are assuming that x >= -1. The final activation in the
                    # previous layer should be a function that satisfies this
                    # (e.g. sigmoid, tanh, relu).
                    average_value = tf.reduce_mean(
                        (x + 1) * tf.expand_dims(
                            tf.cast(mask, tf.float32), axis=-1),
                        axis=1) - 1
                    return average_value

                if flanking_averages and n_flank_length > 0:
                    # Also include average pooled of flanking sequences
                    pooled_flank = Lambda(
                        flanking_extractor, name="%s_extracted" % flank)([
                            convolutional_result,
                            peptide_max_length
                    ])
                    dense_flank = Dense(1,
                        activation="relu", name="%s_avg_dense" % flank)(
                        pooled_flank)

            else:
                assert flank == "c_flank"

                def cleavage_extractor(x):
                    indexer = peptide_max_length + n_flank_length - 1
                    return x[:, indexer]

                single_output_at_cleavage_position = Lambda(
                    cleavage_extractor, name="%s_cleaved" % flank)(
                        single_output_result)

                def max_pool_over_peptide_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst

                    # We generate a per-sample mask that is 1 for all peptide
                    # positions except the last position, and 0 for all other
                    # positions (i.e. n flank, c flank, and the last peptide
                    # position).
                    starts = n_flank_length
                    limits = n_flank_length + peptide_length - 1
                    row = tf.expand_dims(tf.range(0, x.shape[1]), axis=0)
                    mask = tf.logical_and(
                        tf.greater_equal(row, starts),
                        tf.less(row, limits))

                    # We are assuming that x >= -1. The final activation in the
                    # previous layer should be a function that satisfies this
                    # (e.g. sigmoid, tanh, relu).
                    max_value = tf.reduce_max(
                        (x + 1) * tf.expand_dims(
                            tf.cast(mask, tf.float32), axis=-1),
                        axis=1) - 1

                    # We flip the sign so that initializing the final dense
                    # layer weights to 1s is reasonable.
                    return -1 * max_value

                max_over_peptide = Lambda(
                    max_pool_over_peptide_extractor,
                    name="%s_internal_cleaved" % flank)([
                        single_output_result,
                        peptide_max_length
                    ])

                def flanking_extractor(lst):
                    import tensorflow as tf
                    (x, peptide_length) = lst

                    # mask is 1 for c_flank positions and 0 elsewhere.
                    starts = n_flank_length + peptide_length
                    limits = n_flank_length + peptide_length + c_flank_length
                    row = tf.expand_dims(tf.range(0, x.shape[1]), axis=0)
                    mask = tf.logical_and(
                        tf.greater_equal(row, starts),
                        tf.less(row, limits))

                    # We are assuming that x >= -1. The final activation in the
                    # previous layer should be a function that satisfies this
                    # (e.g. sigmoid, tanh, relu).
                    average_value = tf.reduce_mean(
                        (x + 1) * tf.expand_dims(
                            tf.cast(mask, tf.float32), axis=-1),
                        axis=1) - 1
                    return average_value

                if flanking_averages and c_flank_length > 0:
                    # Also include average pooled of flanking sequences
                    pooled_flank = Lambda(
                        flanking_extractor, name="%s_extracted" % flank)([
                            convolutional_result,
                            peptide_max_length
                    ])
                    dense_flank = Dense(1,
                        activation="relu", name="%s_avg_dense" % flank)(
                        pooled_flank)

            outputs_for_final_dense.append(single_output_at_cleavage_position)
            outputs_for_final_dense.append(max_over_peptide)
            if dense_flank is not None:
                outputs_for_final_dense.append(dense_flank)
         
        pep_length = Dense(
            1,
            activation='relu',
            name='peptide_length_dense')(model_inputs['peptide_length'])
        outputs_for_final_dense.append(pep_length)
               
        if len(outputs_for_final_dense) > 1:
            concatenated = Concatenate(name='final')(outputs_for_final_dense)
        else:
            concatenated = outputs_for_final_dense[0]
        
        
        output = Dense(
            post_convolutional_dense_layer_sizes,
            activation="relu",
            name='output_dense1')(concatenated)
        output = Dropout(name="output_dropout1",
                         rate=dropout_rate)(output)
        output = Dense(
            post_convolutional_dense_layer_sizes,
            activation="relu",
            name='output_dense2')(output)
        output = Dropout(name="output_dropout2",
                         rate=dropout_rate)(output)           
        output = Dense(
            1,
            activation="sigmoid",
            name="output_final",
            kernel_initializer=initializers.Ones(),
            )(output)
        
        model = Model(
            inputs=[model_inputs[name] for name in sorted(model_inputs)],
            outputs=[output],
            name="predictor")

        return model

    def __getstate__(self):
        """
        serialize to a dict. Model weights are included. For pickle support.

        Returns
        -------
        dict

        """
        self.update_network_description()
        result = dict(self.__dict__)
        result['_network'] = None
        return result

    def __setstate__(self, state):
        """
        Deserialize. For pickle support.
        """
        self.__dict__.update(state)

    def get_weights(self):
        """
        Get the network weights

        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        self.update_network_description()
        return self.network_weights

    def get_config(self):
        """
        serialize to a dict all attributes except model weights

        Returns
        -------
        dict
        """
        self.update_network_description()
        result = dict(self.__dict__)
        del result['_network']
        result['network_weights'] = None
        return result

    @classmethod
    def from_config(cls, config, weights=None):
        """
        deserialize from a dict returned by get_config().

        Parameters
        ----------
        config : dict
        weights : list of array, optional
            Network weights to restore

        Returns
        -------
        Class1ProcessingNeuralNetwork
        """
        config = dict(config)
        instance = cls(**config.pop('hyperparameters'))
        instance.__dict__.update(config)
        instance.network_weights = weights
        assert instance._network is None
        return instance
