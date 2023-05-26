Table of Contents
=====================
 * [MHCrank](#mhcrank)
     * [Downloading MHCrank](#downloading-mhcrank)
     * [Implementation](#implementation)
     * [Citing](#citing)
     * [Credits and Contact Information](#credits--contact-information)
     * [Copyright and License Notice](#copyright--license-notice)



# MHCrank
[MHC I](https://en.wikipedia.org/wiki/MHC_class_I) processing
prediction model.

MHCrank implements class I peptide processing prediction. 
MHCrank runs on Python 3.4+ using the
[tensorflow](https://www.tensorflow.org/) neural network library.

MHCrank models the selection of digested peptides to be selected to be presented by the MHC Class I molecule and predicts the likelihood that a peptide will be processed for presentation. Predictions are HLa-independent. Models are tested and trained on mass spec-identified antigens / peptides and *in silico* derived decoys. 


## Downloading MHCrank
```bash
git clone https://github.com/ninglab/mhcrank.git
```


## Implementation
Here is the procedure to try out MHCrank on the sample datasets provided using the command line interface

### Train MHCrank

#### Define Hyperparamters:
There are other hyperparameters that may be adjusted, but the most important are listed below. 

For exhaustive list, use command `python mhc_rank/make_architectures.py -h`

```bash
python mhc_rank/make_architectures.py --output </output/directory/for/models/>
                                      --learn_embedding <embedding methods (space delimited): no (for BLOSUM62), yes, and/or combo>
                                      --learned_embed_dims <space demlimited list of ints to try for amino acid learned embedding vector>
                                      --peptide_max_length <space demlimited list of ints describing length to process peptides to try>
                                      --cleave_radius <space demlimited list of ints describing radius of cleavage site for CSSK to try>
                                      --conv_n_kernels <space demlimited list of ints describing number of filters for initial conv layer to try>
                                      --conv_kernel_size <space demlimited list of ints describing kernel size to try>
                                      --dense_layer_size <space demlimited list of ints describing number of units in dense layer to try>
```
The above command will prodice a file ('hyperparameters.json') in the specified output directory that possess all combinations of the supplied hyperparameters.

#### Initialize Training:
```bash
qsub train_APmodels_init.sh data/training_data.csv data/pre_fold.csv </output/directory/for/models/>hyperparameters.json </output/directory/for/models/>
```
This produces the following files in the supplied output directory: 'info.txt', 'manifest.csv', 'train_data.csv.bz2', and 'training_init_info.pkl'. These are used during training and will allow for training process to pick up where it was if it were to be interuppted.

#### Train Models:
```bash
qsub train_APmodels_qsub.sh </output/directory/for/models/>
```

#### Get Training Results:
```bash
python get_training_results.py --num_archs <number of hyperparameter combinations>
                               --num_folds 4
                               --out </output/directory/for/models/>
```

### Get Predictions
#### Create an ensemble:
If desired, an ensemble can be created using the following jupyter notebook: `move_selected_models.ipynb`
Note that the directories will need to be changed according to your implementation.

#### Predict using selected models:
```bash
qsub start_pred.sh <prediction_outfile> <model_name> data/testing_data.csv.gz 00 <weight number> 0 50000
```

#### Get Statistics:
Statistics on predictions can be obtained via the following jupyter notebook: `get_statistics.ipynb`
Note that the directories and file names will need to be changed according to your implementation.


## Citing
Our manuscript was published in Cell Reports Methods and is available
[here](10.1016/j.crmeth.2022.100293).

If you find MHCrank useful in your research, please cite it using the following BibTex entry:
```
@article{lawrence_mhcrank_2021,
  title = {Improving MHC Class I antigen-processing predictions using representation learning and 
           cleavage site-specific kernels},
  journal = {Cell Rep Methods},
  month = {09},
  year = {2022},
  doi = {10.1016/j.crmeth.2022.100293},
  author = {Lawrence, Patrick J and Ning, Xia},
}
```

If you use any part of this library in your research, please cite it using the following BibTex entry:
```
@online{mhcrank,
  title = {MHCrank library for predicting MHC Class I peptide processing},
  author = {Lawrence, Patrick J and Ning, Xia},
  url = {https://github.com/ninglab/mhcrank},
  year = {2021},
}
```



## Credits & Contact Information
This implementation of MHCrank was written by Patrick J. Lawrence with contributiuons by Xia Ning, PhD

If you have questions or encounter a problem, please contact Patrick J. Lawrence at <a href='mailto:patrick.skillman-lawrence@osumc.edu'>patrick.skillman-lawrence@osumc.edu</a>


## Copyright & License Notice
Copyright 2021, The Ohio State University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
