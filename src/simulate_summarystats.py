import time
import subprocess
import sys
import os
import random
import math
import glob
import re

import pandas as pd
import numpy as np
import tensorflow as tf

from multiprocessing import Pool

from astroML.density_estimation import EmpiricalDistribution

'''
Author: Sebastian Burgstaller-Muehlbacher
A Python module which takes a table of simulated trees and generates multiple sequence alignments (MSAs). 
After generating these, it samples x times from it, with mask dimensions of block_taxa by block_length and concatenates
the samples to yield a matrix of dimensions block_length by (block_taxa * samples).

It also converts the nucleotides into a 4 channel tensor, representing nucleotide positions as 1 and non-nucleotide 
position as 0 in a nucleotides channel.
'''

# set the paths to seq_gen, simulation parameter file and simulation data target directory.
assert os.getenv('SEQ_GEN_PATH'), 'Path to seq_gen is not set!'
assert os.getenv('SIM_PARAMS_DIR'), 'Path to simulation parameters file missing!'
assert os.getenv('SIM_DATA_TARGET_DIR'), 'Path to simulation data target directory not set!'
assert os.getenv('TRAIN_DATA_DIR'), 'TRAIN_DATA_PATH not set! This is where training data is read from.'

seq_gen_path = os.getenv('SEQ_GEN_PATH')
sim_params_path = os.getenv('SIM_PARAMS_DIR')
trees_df = pd.read_csv(os.path.join(sim_params_path, 'trees.csv.gz'), compression='gzip')
sim_target_dir = os.getenv('SIM_DATA_TARGET_DIR')  # path for raw MSA files
train_data_path = os.getenv('TRAIN_DATA_DIR')
test_data_path = os.getenv('TEST_DATA_DIR')

samples = 500
block_taxa = 4
input_length_lstm = 10000
block_length = 40


model_int_map = {
    'JC': 0,
    'K2P': 1,
    'K80': 1,
    'F81': 2,
    'HKY': 3,
    'TN93': 4,
    'GTR': 5,
    'JC+G': 6,
    'K2P+G': 7,
    'K80+G': 7,
    'F81+G': 8,
    'HKY+G': 9,
    'TN93+G': 10,
    'GTR+G': 11,

    'K3P': 12,
    'AK2P': 13,
    'TNEG': 14,
    'SYM': 15,
}

model_int_map_model_only = {
    'JC': 0,
    'K2P': 1,
    'K80': 1,
    'F81': 2,
    'HKY': 3,
    'TN93': 4,
    'GTR': 5,
    'JC+G': 0,
    'K2P+G': 1,
    'K80+G': 1,
    'F81+G': 2,
    'HKY+G': 3,
    'TN93+G': 4,
    'GTR+G': 5,

    'K3P': 12,
    'AK2P': 13,
    'TNEG': 14,
    'SYM': 15,
}

model_int_map_nonG_vsG = {
    'JC': 0,
    'K2P': 0,
    'K80': 0,
    'F81': 0,
    'HKY': 0,
    'TN93': 0,
    'GTR': 0,
    'JC+G': 1,
    'K2P+G': 1,
    'K80+G': 1,
    'F81+G': 1,
    'HKY+G': 1,
    'TN93+G': 1,
    'GTR+G': 1,
}

models = {
    'JC': 'seq-gen -mGTR -r1,1,1,1,1,1 -f0.25,0.25,0.25,0.25 -l{seq_len}',

    'K2P': 'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f0.25,0.25,0.25,0.25 -l{seq_len}',

    'K80': 'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f0.25,0.25,0.25,0.25 -l{seq_len}',  # K80 is a different string for K2P

    'K3P': 'seq-gen -mGTR -r{rAC},{rAG},1,1,{rAG},{rAC} -f0.25,0.25,0.25,0.25 -l{seq_len}',

    'AK2P': 'seq-gen -mGTR -r1,1,{rAG},{rAG},1,1 -f0.25,0.25,0.25,0.25 -l{seq_len}',

    'F81': 'seq-gen -mGTR -r1,1,1,1,1,1 -f{piA},{piC},{piG},{piT} -l{seq_len}',

    'HKY': 'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f{piA},{piC},{piG},{piT} -l{seq_len}',

    'TN93': 'seq-gen -mGTR -r1,{rAG},1,1,{rCT},1 -f{piA},{piC},{piG},{piT} -l{seq_len}',

    'TNEF': 'seq-gen -mGTR -r1,{rAG},1,1,{rCT},1 -f0.25,0.25,0.25,0.25 -l{seq_len}',


    'GTR': 'seq-gen -mGTR -r{rAC},{rAG},{rAT},{rCG},{rCT},1 -f{piA},{piC},{piG},{piT} -l{seq_len}',

    'SYM': 'seq-gen -mGTR -r{rAC},{rAG},{rAT},{rCG},{rCT},{rGT} -f0.25,0.25,0.25,0.25 -l{seq_len}',

    'JC+G': 'seq-gen -mGTR -r1,1,1,1,1,1 -f0.25,0.25,0.25,0.25 -l{seq_len} -a {alpha}',  # -a parameter for the alpha

    'K2P+G': 'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f0.25,0.25,0.25,0.25 -l{seq_len} -a {alpha}',

    'K80+G': 'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f0.25,0.25,0.25,0.25 -l{seq_len} -a {alpha}',  # K80 is a different string for K2P

    'F81+G': 'seq-gen -mGTR -r1,1,1,1,1,1 -f{piA},{piC},{piG},{piT} -l{seq_len} -a {alpha}',

    'HKY+G': 'seq-gen -mGTR -r1,{rAG},1,1,{rAG},1 -f{piA},{piC},{piG},{piT} -l{seq_len} -a {alpha}',

    'TN93+G': 'seq-gen -mGTR -r1,{rAG},1,1,{rCT},1 -f{piA},{piC},{piG},{piT} -l{seq_len} -a {alpha}',

    'GTR+G': 'seq-gen -mGTR -r{rAC},{rAG},{rAT},{rCG},{rCT},1 -f{piA},{piC},{piG},{piT} -l{seq_len} -a {alpha}',


}

nucleotide_float_map = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

lanfear_params = pd.read_csv(os.path.join(sim_params_path, 'parameters_lanfear.tsv.gz'), compression='gzip', sep='\t')
lanfear_branchlen = pd.read_csv(os.path.join(sim_params_path, 'branchlen_lanfear.tsv.gz'), compression='gzip', sep='\t')

# collect the empirical distributions for all parameters and branch lengths in a dict
lanfear_eds = dict()
for sim_col, param_name in [('AC', 'rAC'), ('AG', 'rAG'), ('AT', 'rAT'), ('CG', 'rCG'), ('CT', 'rCT'), ('GT', 'rGT'), ('fA', 'piA'), ('fC', 'piC'), ('fG', 'piG'), ('fT', 'piT')]:
    
    lanfear_eds[param_name] = EmpiricalDistribution(lanfear_params[sim_col].loc[lanfear_params[sim_col] < 10])

lanfear_eds['external_branchlen'] = EmpiricalDistribution(lanfear_branchlen['branchlen'].loc[lanfear_branchlen.extint == 'e'])
lanfear_eds['internal_branchlen'] = EmpiricalDistribution(lanfear_branchlen['branchlen'].loc[lanfear_branchlen.extint == 'i'])


def redo_tree(tree, msa_length):
    p = re.compile('[^t0-9]:[0-9.e-]*')

    fragments = []
    last_end = 0
    for x in p.finditer(tree):
        r, l = x.span()
        r += 2
        
        bl = 0
        while bl == 0:
            t = lanfear_eds['internal_branchlen'].rvs(1)[0]
            if t > 1/msa_length:
                bl = t
 
        fragment = tree[last_end:r] + f'{bl:.11f}'
        fragments.append(fragment)
        last_end = l
    
    fragments.append(tree[last_end:])
    tree = ''.join(fragments)

    # replace terminal branches
    p = re.compile('t[0-9]*:[0-9.e-]*')
    
    fragments = []
    last_end = 0
    for x in p.finditer(tree):
        r, l = x.span()
    
        taxon_label = tree[r:l].split(':')[0]

        bl = 0
        while bl == 0:
            t = lanfear_eds['external_branchlen'].rvs(1)[0]
            if t > 1 / msa_length:
                bl = t
    
        fragment = tree[last_end:r] + taxon_label + ':' + f'{bl:.11f}'
        fragments.append(fragment)
        last_end = l
        
    fragments.append(tree[last_end:])
    return ''.join(fragments)
    

# write the parameters into a list instead of interating through a pandas DataFrame
trees = [x for c, x in trees_df.iterrows()]
print(len(trees))

benchmark = False


def generate_raw_msa(model):
    """
    Generates a MSA by sampling from the parameter file
    :param model: evolutionary model the MSA should follow
    :return: a list representing each line of the raw alignment
    """
    params = trees[np.random.randint(0, len(trees), size=1, dtype=int)[0]]
    tree = params['treeNewick']

    with subprocess.Popen([(seq_gen_path + models[model]).format(**params) + ' <<< "{}"'.format(tree)],
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          ) as proc:

        return proc.stdout.readlines()


def process_msa_pairwise(lines):
    num_positions = 10000

    def make_profile(m):

        tstv_map = {
            1 << 1: 0,  # 2 AA 
            1 << 3: 14, # 8 AC 
            1 << 5: 15, # 32 AG 
            1 << 7: 16, # 128 AT 
            3 << 1: 20, # 6 CA 
            3 << 3: 1,  # 24 CC 
            3 << 5: 18,  # 96 CG 
            3 << 7: 17, # 384 CT 
            5 << 1: 21, # 10 GA 
            5 << 3: 24, # 40 GC 
            5 << 5: 2, # 160 GG 
            5 << 7: 19, # 640 GT 
            7 << 1: 22, # 14 TA 
            7 << 3: 23, # 56 TC 
            7 << 5: 25, # 224 TG 
            7 << 7: 3, # 896 TT 

        }
        mm = np.zeros([num_positions, 26], dtype=np.float16)
        
        m = m.T

        # replace str array with np.uint8 array
        # MSA integer map
        msa_int_map = {
            'A': np.uint8(1),
            'C': np.uint8(3),
            'G': np.uint8(5),
            'T': np.uint8(7),
        }
        d = np.zeros(m.shape, dtype=np.uint16)

        for k, v in msa_int_map.items():
            d[m == k] = v  

        m = d

        for c in range(0, num_positions):

            # make sure that two different strands are being selected
            while True:
                first_strand = np.random.randint(0, m.shape[1], size=1, dtype=np.int)[0]
                second_strand = np.random.randint(0, m.shape[1], size=1, dtype=np.int)[0]
                
                if first_strand != second_strand:
                    break

            fsb, first_strand_base_counts = np.unique(m[:, first_strand], return_counts=True)
            if fsb.shape[0] == 4:
                mm[c, 4:8] = first_strand_base_counts / msa_length
            else:  # handle the (rarer) case where there are fewer than 4 bases in a sequence
                bm = {1: 4, 3: 5, 5: 6, 7: 7}
                for bc, base in enumerate(fsb):
                    mm[c, bm[base]] = first_strand_base_counts[bc] / msa_length

            ssb, second_strand_base_counts = np.unique(m[:, second_strand], return_counts=True)
            if ssb.shape[0] == 4:
                mm[c, 8:12] = second_strand_base_counts / msa_length
            else:
                bm = {1: 8, 3: 9, 5: 10, 7: 11}
                for bc, base in enumerate(ssb):
                    mm[c, bm[base]] = second_strand_base_counts[bc] / msa_length
            
            tstvs, tstv_counts = np.unique(m[:, first_strand] << m[:, second_strand], return_counts=True)

            for cc, t in enumerate(tstvs):
                ind = tstv_map[t]
                mm[c, ind] = tstv_counts[cc] / msa_length

            # count transitions
            transversion_counts = 0
            for t in [1 << 3, 3 << 1, 1 << 7, 7 << 1, 3 << 5, 5 << 3, 5 << 7, 7 << 5]:

                occ = np.where(tstvs == t)[0]
                # make sure the specific transversion exists in a certain MSA
                if occ.shape[0] > 0:
                    transversion_counts += tstv_counts[occ[0]]
            mm[c, 12] = transversion_counts / msa_length

            transition_counts = 0
            for t in [1 << 5, 5 << 1, 3 << 7, 7 << 3]:

                occ = np.where(tstvs == t)[0]
                # make sure the specific transversion exists in a certain MSA
                if occ.shape[0] > 0:
                    transition_counts += tstv_counts[occ[0]]
            mm[c, 13] = transition_counts / msa_length

        return mm

    raw_msa = np.array([x.decode().strip().split(' ')[-1] for x in lines[1:]])

    msa = np.array([list(y) for y in raw_msa])

    del raw_msa

    try:
        msa_height, msa_width = msa.shape
    except ValueError as e:
        print(e)
        print('MSA shape:', msa.shape)

    msa_length = msa.shape[1]
    msa = make_profile(msa)
    msa = np.reshape(msa, (40, 250, 26))

    return msa


def process_msa_tstv(lines):
    num_positions = input_length_lstm
    def make_profile(m):
        # print(m.shape)
        mm = np.zeros([num_positions, 14], dtype=np.float16)
        for c in range(0, num_positions):

            tstv_map = [
                'AC',
                'AG',
                'AT',
                'CT',
                'CG',
                'GT',
                'AA',
                'CC',
                'GG',
                'TT',

              #  'CA',
              #  'GA',
              #  'TA',
              #  'TC',
              #  'GC',
              #  'TG',
            ]

            base_map = {
                'A': 0,
                'C': 0,
                'G': 0,
                'T': 0
            }
            for z in m[:, c]:
                base_map[z] += 1

            for count, tt in enumerate(tstv_map):
                first, second = tt[0], tt[1]

                if first == second and base_map[first] > 1:
                    mm[c, count + 4] = base_map[first] / m.shape[0]
                elif first != second:
                    mm[c, count + 4] = base_map[first] * base_map[second] / m.shape[0]

            # print(base_map)

            for k, v in nucleotide_float_map.items():
                # print(v, c)
                mm[c, v] = float(base_map[k]) / m.shape[0]

        return mm


    raw_msa = np.array([x.decode().strip().split(' ')[-1] for x in lines[1:]])

    msa = np.array([list(y) for y in raw_msa])

    del raw_msa

    try:

        msa_height, msa_width = msa.shape
    except ValueError as e:
        print(e)
        print('MSA shape:', msa.shape)

    idx = np.random.randint(0, msa.shape[1], size=num_positions, dtype=np.int)
    msa = msa[:, idx]

    msa = make_profile(msa)
    msa = np.reshape(msa, (40, 250, 14))

    return msa


def process_msa_conv(lines):
    benchmark = False

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    raw_msa = np.array([x.decode().strip().split(' ')[-1] for x in lines[1:]])
    msa = np.array([list(y) for y in raw_msa])

    try:

        msa_height, msa_width = msa.shape
    except ValueError as e:
        print(e, 'error when processing subsamples MSA')
        print('MSA shape:', msa.shape)

    sub_samples = []

    if benchmark:
        print('duration raw alignment creation:', (time.time() - start_time) / 60)

    start_time = time.time()

    for sample_count in range(samples):
        rand_taxa = np.random.randint(0, msa.shape[0], size=block_taxa, dtype=np.int)
        rand_seq_start = np.random.randint(0, msa.shape[1], size=1, dtype=np.int)[0]

        if rand_seq_start + block_length >= msa_width:
            overflow_size = block_length - (msa_width - rand_seq_start)

            smpl = np.concatenate(
                (msa[rand_taxa, rand_seq_start:msa_width], msa[rand_taxa, 0:overflow_size]),
                axis=1)
            sub_samples.append(smpl)
        else:
            smpl = msa[rand_taxa, rand_seq_start:(rand_seq_start + block_length), ]
            sub_samples.append(smpl)

    if benchmark:
        print('duration subsampling:', (time.time() - start_time) / 60)
    start_time = time.time()

    sampled_msa = np.transpose(np.concatenate(sub_samples, axis=0))

    msa = np.zeros([len(sampled_msa), len(sampled_msa[0]), 4], dtype=np.int8)
    for base, channel in nucleotide_float_map.items():
        base_msa = (sampled_msa == base) * 1

        msa[:, :, channel] = base_msa

    return msa, msa_height, msa_width


def process_msa_frequencies(lines):
    freq_block_length = 10000
    def make_profile(m):

        mm = np.zeros([ freq_block_length, 4], dtype=np.float16)
        for c in range(0, freq_block_length):

            base_map = {
                'A': 0,
                'C': 0,
                'G': 0,
                'T': 0
            }
            for z in m[:, c]:
                base_map[z] += 1

            for k, v in nucleotide_float_map.items():
                mm[ c, v] = float(base_map[k]) / m.shape[0]

        return mm

    raw_msa = np.array([x.decode().strip().split(' ')[-1] for x in lines[1:]])

    msa = np.array([list(y) for y in raw_msa])

    del raw_msa

    try:
        msa_height, msa_width = msa.shape
    except ValueError as e:
        print(e)
        print(msa)

    idx = np.random.randint(0, msa.shape[1], size=freq_block_length, dtype=np.int)
    msa = msa[:, idx]

    msa = make_profile(msa)

    return msa


def generate_msa(t):
    """
    The worker function for multi-threaded MSA generation and preprocessing
    :param t: A tuple containing all required parameters:
        model: Model string
        num_msas: Number of MSAs to generate
        save_msa: Whether or not to save the MSA file to disk
        potential_alphas: A list of alpha parameters for the Gamma distribution
        seq_len: The sequence length of the MSA to simulate
        nr_taxa: The number of taxa which should be in the MSA
    :return: A List of tuples (<Evolutionary model string>, <pre-processed MSA numpy array>)
    """
    # tmp = []
    model, save_msa, potential_alphas, seq_len, nr_taxa = t

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    # A map for the model parameter file, so the parameters for the correct number of taxa can be selected.
    mp = {8: 0, 16: 1, 32: 2, 64: 3, 128: 4}

    # print(model, num_msas)
    # for x in range(num_msas):
    alpha = 0
    # np.random.shuffle(potential_alphas)

    if model[-2:] == '+G':
        # list of potential alphas has been shuffled, so first element can be taken als the alpha parameter
        alpha = potential_alphas

    # the while loop serves to draw new parameters and simulate a new MSA in the rare case simulation failes with lanfear params
    while True:
        param_start, param_stop = mp[nr_taxa] * 10000, (mp[nr_taxa] + 1) * 10000
        idx = np.random.randint(param_start, param_stop, size=1, dtype=np.int)[0]
        # print(idx)
        params = trees[idx]
        tree = redo_tree(params['treeNewick'], seq_len)

        start_time = time.time()
        rates_freq = {}
        for k, v in lanfear_eds.items():
            rates_freq.update({k: v.rvs(1)[0]})

        cmd = (seq_gen_path + models[model]).format(**rates_freq, seq_len=seq_len, alpha=alpha) + ' <<< "{}"'.format(tree)
        # print(cmd.split('/')[-1])

        # Call seq-gen, read in generated MSA.
        with subprocess.Popen([cmd],
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              ) as proc:
            lines = proc.stdout.readlines()
            if len(lines) - 1 != nr_taxa:
                print(cmd)
                print(nr_taxa, tree)
                print(proc.stderr.readlines())  # can be used to print seq-gen console output.
                continue

            # generate a secondary seed
            fn_rand_salt = np.random.randint(0, 99999, size=1, dtype=int)[0]

            # make sure that alpha parameter value gets added to filename
            alpha_str = ''
            if alpha > 0:
                alpha_str = '_alpha{}_'.format(alpha)

            # Save the MSA to a file if save_msa flag is set.
            if save_msa:

                filename = '{seed}_{nr_taxa}Taxa_{model}_{alpha}_{seq_len}_{fn_rand_salt}.phy'\
                    .format(seed=params['seed'],
                            nr_taxa=nr_taxa, model=model,
                            alpha=alpha_str,
                            seq_len=seq_len,
                            fn_rand_salt=fn_rand_salt)

                path = os.path.join(sim_target_dir, str(seq_len) + 'bp', model)
                full_path = os.path.join(path, filename)
                print(full_path)

                os.makedirs(path, exist_ok=True)
                with open(full_path, 'w') as f:
                    f.writelines([x.decode() for x in lines])

                save_tree = True
                if save_tree:
                    treefile = filename.split('.phy')[0] + '.tree'
                    with open(os.path.join(path, treefile), 'w') as f:
                        f.writelines([tree])
        # get out of the while loop if sucessfully reaching end
        break

    msa_conv,  taxa, msa_length = process_msa_conv(lines)

    # process alignment, append tuple of model string and alignment to tmp list.
    tmp = (model, params['seed'], alpha_str.strip('_'), seq_len, fn_rand_salt, nr_taxa, msa_conv, process_msa_tstv(lines), process_msa_pairwise(lines), process_msa_frequencies(lines))

    if benchmark:
        print('duration raw alignment creation:', (time.time() - start_time) / 60)

    return tmp


def load_msas_thread(file_names):
    """
    The worker for loading and pre-processing MSA files. The filename needs to be processed as it contains information
    which allow to map back to the simulation parameters file. Specifically, there is a primary seed which identifies
    the line in the
    :param file_names:
    :return:
    """
    assert type(file_names) in (list, tuple)

    tmp = []
    for file_name in file_names:
        splits = file_name.split('/')[-1].split('_')
        # print(splits)
        alpha = "0"
        if len(splits) == 6:
            seed, taxa, model, _, seq_len, sec_seed = splits
        else:
            seed, taxa, model, _, alpha, _, seq_len, sec_seed = splits
        with open(file_name, 'rb') as f:
            lines = f.readlines()
            msa_conv,  taxa, msa_length = process_msa_conv(lines)
            tmp.append((model, seed, alpha, seq_len, sec_seed.split('.')[0], msa_length, taxa, msa_conv,  process_msa_tstv(lines), process_msa_pairwise(lines), process_msa_frequencies(lines)))
    return tmp


def _int64_feature(value):
    """
    A function which generates an Int64 .tfrecords feature.
    :param value: The value which should be converted.
    :return: An Int64 tfrecords value.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
    A function which generates a bytes .tfrecords feature.
    :param value: The value which should be converted.
    :return: A bytes tfrecords value.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """
    A function which generates a fload .tfrecord feature.
    :param value: The value which should be converted.
    :return: A bytes tfrecord value.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def sim_msas(msas_per_model, models, num_threads, training_data=True, taxa=(8,),
             alphas=[0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5], model_int_map=model_int_map, seq_len=1000, return_seeds=False,
             data_dir=train_data_path, save_msa=False, tfrecord_file_prefix='msa_data_{}', examples_per_tfr_file=500):
    """ A function which simulates Multiple Sequence Aligments, in a multi-threaded fashion. Important: the total number
    of MSAs simulated is msas_per_model * len(models) * len(taxa)
    :param msas_per_model: How many MSAs should be generated per model in the models parameter.
    Equal numbers will be created for each taxa and alphas parameter.
    Also, a minimum number of MSAs will be created, determined by len(alphas) * len(taxa).
    :type msas_per_model: int
    :param models: A list of model names, specified in the models dict or model_int_map dict above.
    :type: models: list of str
    :param num_threads: Number of threads to be used for simulation
    :type num_threads: int
    :param training_data: Whether training data are being simulated and thus these should be written to a .tfrecords
    file.
    :param training_data: bool
    :param taxa: A list specifying the number of taxa for an MSA to be simulated. Taxa must match parameter file.
    :type taxa: iterable of int
    :param model_int_map: A dictionary which maps model string representations to integer representations
    :type model_int_map: dict
    :return: None
    """

    start_time = time.time()
    print(model_int_map)

    alphas_per_model = math.ceil(msas_per_model / (len(alphas) * len(taxa))) * alphas

    work_packages = [(model, save_msa, alpha, seq_len, t) for model in models for t in taxa
                     for alpha in alphas_per_model]
    required_file_count = math.ceil(len(work_packages) / examples_per_tfr_file)

    print("Work package size:", len(work_packages))
    print("Files to be generated:", required_file_count)

    random.shuffle(work_packages)
    print(work_packages[:50])

    # required_file_count = math.ceil(len(models) * tasks * num_threads / 10000)
    param_sets_per_file = math.ceil(len(work_packages) / required_file_count)

    for f in range(required_file_count):
        sub = work_packages[f * param_sets_per_file:(f + 1) * param_sets_per_file]
        print('Generating tfrecord file #', f + 1, 'of', required_file_count)

        with Pool(num_threads) as p:
            test = p.imap_unordered(generate_msa, sub, 5)

            msa_list = [y for y in test]
        random.shuffle(msa_list)

        tstv_data = np.zeros([len(msa_list), 40, 250, 14], dtype=np.float16)
        pairwise_data = np.zeros([len(msa_list), 40, 250, 26], dtype=np.float16)
        conv_data = np.zeros([len(msa_list), block_length, samples * block_taxa, 4], dtype=np.int8)
        freq_data = np.zeros([len(msa_list), 10000, 4], dtype=np.float16)
        labels = np.zeros([len(msa_list)], dtype=np.int8)
        seeds = np.zeros([len(msa_list)], dtype=np.int64)
        sec_seeds = np.zeros([len(msa_list)], dtype=np.int64)
        alphas = np.zeros([len(msa_list)], dtype=np.float32)
        ev_model_strings = []

        lengths = np.zeros([len(msa_list)], dtype=np.int64)
        taxa = np.zeros([len(msa_list)], dtype=np.int64)

        for count, (label, seed, alpha, seq_len, sec_seed, taxa_count, conv_msa, tstv_msa, pairwise_msa, freq_msa) in enumerate(msa_list):
            conv_data[count, :, :] = conv_msa.astype(np.int8)
            tstv_data[count, :, :] = tstv_msa
            pairwise_data[count, :, :] = pairwise_msa
            freq_data[count, :, :] = freq_msa
            labels[count] = model_int_map[label]
            seeds[count] = int(seed)
            sec_seeds[count] = int(sec_seed)
            alphas[count] = float(alpha.split('alpha')[-1]) if alpha else 50
            ev_model_strings.append(label)

            lengths[count] = seq_len
            taxa[count] = taxa_count

        labels = np.eye(12, dtype=np.int8)[labels.reshape(-1)]

        print(labels.shape)

        if training_data:
            # make sure tfrecords data dir exists
            os.makedirs(data_dir, exist_ok=True)

            # location to save the TFRecords file
            train_filename = os.path.join(data_dir, tfrecord_file_prefix + '_{}'.format(f) + '.tfrecords')
            # open the TFRecords file
            writer = tf.io.TFRecordWriter(train_filename)
            for i in range(len(conv_data)):
                # print how many images are saved every 1000 images
                if not i % 1000:
                    print('Train data: {}/{}'.format(i, len(conv_data)))
                    sys.stdout.flush()

                # Create a feature
                feature = {
                    'train/label': _bytes_feature(labels[i].tobytes()),
                    'train/conv_msa': _bytes_feature(conv_data[i].tobytes()),
                    'train/tstv_msa': _bytes_feature(tstv_data[i].tobytes()),
                    'train/freq_msa': _bytes_feature(freq_data[i].tobytes()),
                    'pairwise__msa': _bytes_feature(pairwise_data[i].tobytes()),
                    'train/seed': _int64_feature(seeds[i]),
                    'train/sec_seed': _int64_feature(sec_seeds[i]),
                    'train/alpha': _float_feature(alphas[i]),
                    'train/ev_model': _bytes_feature(ev_model_strings[i].encode()),

                    'train/seq_lengths': _int64_feature(lengths[i]),
                    'train/taxa': _int64_feature(taxa[i]),
                
                }
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            writer.close()
            sys.stdout.flush()

    return None, None


def load_msas(filenames, num_threads, create_tfrecord=True, data_dir=train_data_path, tfrecord_file_prefix='msa_data_',
              model_int_map=model_int_map, msas_per_file=10000):
    """ A function which loads alignments from files instead of simulating new ones
    :param filenames: A list of file names which should be loaded
    :param num_threads: The number of threads which should be used for preprocessing the files.
    :return: A numpy.array with training/testing data and a numpy.array with training/testing labels.
    """

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    required_file_count = math.ceil(len(filenames) / msas_per_file)
    print('Will generate {} tfrecord files'.format(required_file_count))

    for f in range(required_file_count):
        sub_fn = filenames[f * msas_per_file:(f + 1) * msas_per_file]
        tasks = math.ceil(len(sub_fn) / num_threads)

        param = (list(chunks(sub_fn, tasks)))
        print('Generating tfrecord file #', f)

        with Pool(num_threads) as p:
            test = p.map(load_msas_thread, param)

        msa_list = [x for y in test for x in y]
        # print(msa_list[0])
        random.shuffle(msa_list)

        tstv_data = np.zeros([len(msa_list), 40, 250, 14], dtype=np.float16)
        pairwise_data = np.zeros([len(msa_list), 40, 250, 26], dtype=np.float16)
        conv_data = np.zeros([len(msa_list), block_length, samples * block_taxa, 4], dtype=np.int8)
        freq_data = np.zeros([len(msa_list), 10000, 4], dtype=np.float16)
        labels = np.zeros([len(msa_list)], dtype=np.int8)
        seeds = np.zeros([len(msa_list)], dtype=np.int64)
        sec_seeds = np.zeros([len(msa_list)], dtype=np.int64)
        alphas = np.zeros([len(msa_list)], dtype=np.float)
        ev_model_strings = []
        lengths = np.zeros([len(msa_list)], dtype=np.int64)
        taxa = np.zeros([len(msa_list)], dtype=np.int64)

        for count, (label, seed, alpha, seq_len, sec_seed, length, taxa_count,  conv_msa, tstv_msa, pairwise_msa, freq_msa) in enumerate(msa_list):
            conv_data[count,  :, :] = conv_msa.astype(np.int8)
            tstv_data[count,  :, :] = tstv_msa
            pairwise_data[count,  :, :] = pairwise_msa
            freq_data[count,  :, :] = freq_msa
            labels[count] = model_int_map[label]
            seeds[count] = int(seed)
            sec_seeds[count] = int(sec_seed)
            alphas[count] = float(alpha.split('alpha')[-1]) if alpha else 50
            ev_model_strings.append(label)
            lengths[count] = length
            taxa[count] = taxa_count

        labels = np.eye(12, dtype=np.int8)[labels.reshape(-1)]

        if create_tfrecord:
            # make sure tfrecords data dir exists
            os.makedirs(data_dir, exist_ok=True)

            train_filename = data_dir + '{}_{}.tfrecords'.format(tfrecord_file_prefix, f)  # address to save the TFRecords file
            # open the TFRecords file
            writer = tf.io.TFRecordWriter(train_filename)
            for i in range(len(conv_data)):
                # print how many images are saved every 1000 images
                if not i % 1000:
                    print('Train data: {}/{}'.format(i, len(conv_data)))
                    sys.stdout.flush()
                
                feature = {
                    'train/label': _bytes_feature(labels[i].tobytes()),
                    'train/conv_msa': _bytes_feature(conv_data[i].tobytes()),
                    'train/tstv_msa': _bytes_feature(tstv_data[i].tobytes()),
                    'train/freq_msa': _bytes_feature(freq_data[i].tobytes()),
                    'pairwise__msa': _bytes_feature(pairwise_data[i].tobytes()),
                    'train/seed': _int64_feature(seeds[i]),
                    'train/sec_seed': _int64_feature(sec_seeds[i]),
                    'train/alpha': _float_feature(alphas[i]),
                    'train/ev_model': _bytes_feature(ev_model_strings[i].encode()),
                    'train/seq_lengths': _int64_feature(lengths[i]),
                    'train/taxa': _int64_feature(taxa[i]),
                }

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            writer.close()
            sys.stdout.flush()

        print(labels.shape)


def main():

    start = time.time()

    ev_models = ['JC', 'K2P', 'F81', 'HKY', 'TN93', 'GTR', 'JC+G', 'K2P+G', 'F81+G', 'HKY+G', 'TN93+G', 'GTR+G']
    alphas = [0.001, 0.01, 0.05, 0.1, 0.3,  0.5, 0.7,  *list(range(1, 11))]

    # simulation example for 1kbp MSAs
    sim_msas(20, ev_models, 10, save_msa=False, alphas=alphas, taxa=(8, 16, 64, 128), seq_len=1000, tfrecord_file_prefix='train_msas_1K', data_dir='./train_data_dir/', model_int_map=model_int_map)
    # sim_msas(250, ev_models, 64, save_msa=False, alphas=alphas, num_taxa=(8, 16, 64, 128), seq_len=1000,  tfrecord_file_prefix='test_msas_1K', data_dir='./test_data_dir/', model_int_map=model_int_map)

    print('duration:', time.time() - start)


if __name__ == '__main__':
    sys.exit(main())



