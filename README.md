# ModelRevelator

## Abstract/Introduction
Selecting the best model of sequence evolution for a multiple-sequence-alignment (MSA) constitutes the first step of phylogenetic tree reconstruction. Common approaches for inferring nucleotide models typically apply maximum likelihood (ML) methods, with discrimination between models determined by one of several information criteria. This requires tree reconstruction and optimisation which can be computationally expensive. We demonstrate that neural networks can be used to perform model selection, without the need to reconstruct trees, optimise parameters, or calculate likelihoods. 

We introduce ModelRevelator, a model selection tool underpinned by two neural networks. The first neural network, NNmodelfind, recommends one of six commonly used models of sequence evolution, ranging in complexity from JC to GTR. The second, NNalphafind, recommends whether or not a Γ--distributed rate heterogeneous model should be incorporated, and if so, provides an estimate of the shape parameter, ɑ. Users can simply input an MSA into ModelRevelator, and swiftly receive output recommending the evolutionary model, inclusive of the presence or absence of rate heterogeneity, and an estimate of ɑ. 

We show that ModelRevelator performs comparably with likelihood-based methods over a wide range of parameter settings, with significant potential savings in computational effort. Further, we show that this performance is not restricted to the alignments on which the networks were trained, but is maintained even on unseen empirical data. ModelRevelator will soon be made freely available in the forthcoming version of IQ-Tree (http://www.iqtree.org), and we expect it will provide a valuable alternative for phylogeneticists, especially where traditional methods of model selection are computationally prohibitive.


## ModelRevelator GitHub Repository
Here, we provide 
* Branch lengths, rate and frequency parameters estimated from the Lanfear Dataset.
* Random trees for 8, 16, 64 and 128 taxa.
* A simulation script to simulate training and test datasets.
* The Tensorflow implementations for training of NNalphafind and NNmodelfind.
* The trained NNmodelfind and NNalphafind neural networks in ONNX format.
* A Python command line tool to infer the model and alpha parameter of your MSA of choice (ModelRevelator CLI).
* Four alignments for testing purposes (example_alignments). File names contain ground truth.

## ModelRevelator Installation
```bash
    git clone https://github.com/Cibiv/ModelRevelator
    
    cd ModelRevelator
    
    pip install -r requirements.txt
```


## ModelRevelator CLI
ModelRevelator CLI is a Python script which integrates NNmodelfind and NNalphafind estimates, 
performed either on single alignments or batches of alignment files. The alignment files need to be in phylip format (.phy).

Commandline options:  
```
    -a OR "--alignments": Alignment file(s) to process. In phylip or fasta format (.phy or .fasta). 
            You can use * or ? placeholders in your path- and filenames to process 
            entire directories containing alignments.
    -r OR "--results_filename": File name you want your results to be stored in. 
            Currently returns a .csv file with all results. Defaults to "modelrevelator_output.csv".
```
Example:  
```
    python modelrevelator_cli.py -a /home/my_user/cool_msa.phy -r my_results.csv 
```

