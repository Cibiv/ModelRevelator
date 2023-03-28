__author__ = 'Sebastian Burgstaller-Muehlbacher'
__copyright__ = "Copyright 2022, Sebastian Burgstaller-Muehlbacher"

import numpy as np
import time
import glob
import pandas as pd

import onnxruntime
from scipy.special import softmax

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-a", "--alignments", help="Alignment file(s) to process. In phylip or fasta format (.phy or .fasta). You can use * or ? placeholders in your path- and filenames to process entire directories containing alignments.")
parser.add_argument("-o", "--output_format", help="Output format: csv", default='csv')
parser.add_argument("-r", "--results_filename", help="File name you want your results to be stored in", default='modelrevelator_output.csv')

args = parser.parse_args()
print("output format:", args.output_format)

results_filename = args.results_filename

benchmark = False
int_model_map = {
    0: 'JC',
    1: 'K2P',
    2: 'F81',
    3: 'HKY',
    4: 'TN93',
    5: 'GTR'
}


def process_msa_pairwise(lines):
    num_sumstats = 10000

    def make_profile(m):
        msa_length = m.shape[1]

        tstv_map = {
            1 << 1: 0,  # 2 AA
            1 << 3: 14,  # 8 AC
            1 << 5: 15,  # 32 AG
            1 << 7: 16,  # 128 AT
            3 << 1: 20,  # 6 CA
            3 << 3: 1,  # 24 CC
            3 << 5: 18,  # 96 CG
            3 << 7: 17,  # 384 CT
            5 << 1: 21,  # 10 GA
            5 << 3: 24,  # 40 GC
            5 << 5: 2,  # 160 GG
            5 << 7: 19,  # 640 GT
            7 << 1: 22,  # 14 TA
            7 << 3: 23,  # 56 TC
            7 << 5: 25,  # 224 TG
            7 << 7: 3,  # 896 TT
        }
        mm = np.zeros([num_sumstats, 26], dtype=np.float16)

        m = m.T

        # replace str array with np.uint8 array
        # MSA integer map
        msa_int_map = {
            'A': np.uint16(1),
            'C': np.uint16(3),
            'G': np.uint16(5),
            'T': np.uint16(7),
        }
        d = np.zeros(m.shape, dtype=np.uint16)

        start = time.time()
        for k, v in msa_int_map.items():
            d[m == k] = v

        m = d

        if benchmark:
            print('Duration MSA remapping', time.time() - start)

        for c in range(0, num_sumstats):

            # With gapped MSAs, we cannot be sure to have sequence pairs of length > 0 after gap removal.
            # So try 1000 times to get such a pair. Else, return zeros.
            try_counter = 0
            while True:
                if try_counter >= 1000:
                    break

                try_counter += 1

                # make sure that two different strands are being selected
                while True:
                    first_strand = np.random.randint(0, m.shape[1], size=1, dtype=int)[0]
                    second_strand = np.random.randint(0, m.shape[1], size=1, dtype=int)[0]

                    if first_strand != second_strand:
                        break

                fs = m[:, first_strand]
                ss = m[:, second_strand]

                # print(ss.shape)

                # For now, remove gaps and degenerate characters
                mask = (fs != 0) & (ss != 0)
                fs = fs[mask]
                ss = ss[mask]

                fs = fs.reshape((fs.shape[0], 1))
                ss = ss.reshape((ss.shape[0], 1))

                # make sure to normalize to new length after removal
                msa_length = len(fs)
                if msa_length <= 0:
                    continue

                # Fist strand base counts calculation
                start = time.time()

                for ccc, x in enumerate(msa_int_map.values()):
                    stat = np.sum(fs == x)
                    mm[c, ccc + 4] = stat / msa_length

                if benchmark:
                    print('Duration FSBC opt:', time.time() - start)

                # Second strand base counts calculation
                start = time.time()

                for ccc, x in enumerate(msa_int_map.values()):
                    stat = np.sum(ss == x)
                    mm[c, ccc + 8] = stat / msa_length

                if benchmark:
                    print('Duration SSBC opt:', time.time() - start)

                # Transition and Transversion counts
                start = time.time()
                t = fs << ss
                tts = [
                    1 << 1,  # 2 AA,
                    1 << 3,  # 8 AC
                    1 << 5,  # 32 AG
                    1 << 7,  # 128 AT
                    3 << 1,  # 6 CA
                    3 << 3,  # 24 CC
                    3 << 5,  # 96 CG
                    3 << 7,  # 384 CT
                    5 << 1,  # 10 GA
                    5 << 3,  # 40 GC
                    5 << 5,  # 160 GG
                    5 << 7,  # 640 GT
                    7 << 1,  # 14 TA
                    7 << 3,  # 56 TC
                    7 << 5,  # 224 TG
                    7 << 7,  # 896 TT
                ]
                for x in [np.uint16(y) for y in tts]:
                    stat = np.sum(t == x)
                    ind = tstv_map[x]
                    mm[c, ind] = stat / msa_length

                if benchmark:
                    print('Duration tstv counts:', time.time() - start)

                # Count transversions only
                start = time.time()
                transversion_counts = 0
                for t in [1 << 3, 3 << 1, 1 << 7, 7 << 1, 3 << 5, 5 << 3, 5 << 7, 7 << 5]:
                    # ['AC', 'CA', 'AT', 'TA', 'CG', 'GC', 'GT', 'TG']:
                    ind = tstv_map[t]
                    transversion_counts += mm[c, ind]

                mm[c, 12] = transversion_counts  # already normalized counts

                if benchmark:
                    print("Duration transversion counts", time.time() - start)

                # Count transitions only
                start = time.time()
                transition_counts = 0
                for t in [1 << 5, 5 << 1, 3 << 7, 7 << 3]:
                    # ['AG', 'GA', 'CT', 'TC']:
                    ind = tstv_map[t]
                    transition_counts += mm[c, ind]

                mm[c, 13] = transition_counts  # already normalized

                if benchmark:
                    print('Duration transition counts', time.time() - start)

                break

        return mm

    msa = np.array([list(y) for y in lines])

    try:
        msa_height, msa_width = msa.shape
    except ValueError as e:
        print(e)
        print('MSA shape invalid:', msa.shape)
        #print(msa)

    msa = make_profile(msa)
    msa = np.reshape(msa, (40, 250, 26))

    return msa


def process_alpha_msa(raw_msa):
    num_sumstats = 10000

    def make_profile(m):
        # print(m.shape)
        mm = np.zeros([num_sumstats, 4], dtype=np.float32)

        arrs = [m == x for x in ['A', 'C', 'G', 'T']]

        for i, x_arr in enumerate(arrs):
            mm[:, i] = np.sum(x_arr, axis=0) / m.shape[0]

        return mm

    msa = np.array([list(y) for y in raw_msa])

    if msa.shape[1] != num_sumstats:
        idx = np.random.randint(0, msa.shape[1], size=num_sumstats, dtype=int)
        msa = msa[:, idx]

    msa = make_profile(msa)
    msa = msa.reshape((1, 10000, 4))

    return msa


# find alignment files to process
msa_files = glob.glob(args.alignments)
print("# of alignments to process:", len(msa_files))

# Load NNmodelfind parameters
ev_model_file = '../trained_nns/nn_modelfind.onnx'
sess_model = onnxruntime.InferenceSession(ev_model_file)

# Load NNalphafind parameters
alpha_model_file = '../trained_nns/nn_alphafind.onnx'
sess_alpha = onnxruntime.InferenceSession(alpha_model_file)

all_results = {
    'file_name': [],
    'alignment_length': [],
    'taxa': [],
    'estimated_model': [],
    'estimated_alpha': [],
    'duration_preprocessing': [],
    'duration_inference_model': [],
    'duration_inference_alpha': [],

}

for count, msa_file in enumerate(msa_files):

    with open(msa_file, 'rb') as f:
        lines = f.readlines()
        proc_start_time = time.time()

        if msa_file.endswith('.phy'):
            raw_msa = np.array([x.decode().strip().split(' ')[-1] for x in lines[1:]])
            # start = time.time()

        elif msa_file.endswith('.fasta'):
            sub_seq = []
            all_lines = []
            for l in lines:
                if l.startswith(b'>'):
                    if len(sub_seq) > 0:
                        all_lines.append(''.join(sub_seq))
                    sub_seq = []
                else:
                    sub_seq.append(l.decode().strip().upper())

            raw_msa = np.array(all_lines)

        proc_model_msa = process_msa_pairwise(raw_msa)
        # print('pairwise prepro duration', time.time() - start)

        # start = time.time()
        proc_alpha_msa = process_alpha_msa(raw_msa)
        # print('columwise preproc duration:', time.time() - start)

    
    proc_end_time = time.time()
    all_results['duration_preprocessing'].append(proc_end_time - proc_start_time) 
    all_results['file_name'].append(msa_file)
    all_results['taxa'].append(len(raw_msa))
    all_results['alignment_length'].append(len(raw_msa[0]))

    # model inference
    ev_model_start_time = time.time()
    proc_model_msa = np.reshape(proc_model_msa.astype(np.float32), [1, 40, 250, 26])
    res = sess_model.run(None, {'input_1': proc_model_msa, 'input_2': np.zeros([1, 40, 250, 14], dtype=np.float32)})
    s_max = softmax(res)
    model_res = np.argmax(s_max)
    ev_model_time = time.time() - ev_model_start_time

    model_string = int_model_map[model_res]
    all_results['duration_inference_model'].append(ev_model_time)
    
    # alpha inference
    alpha_start_time = time.time()
    alpha_res = sess_alpha.run(None, {'input_1': proc_alpha_msa})
    alpha_s_max = softmax(alpha_res[0])
    het_res = np.argmax(alpha_s_max)
    estimated_alpha = alpha_res[1][0][0] / 1000
    alpha_time = time.time() - alpha_start_time

    # Final, combined output for ModelRevelator:
    # add rate heterogeneity to model and provide estimate for alpha parameter, in case NNalphafind says so.
    if het_res == 1:
        all_results['estimated_model'].append(model_string + '+G')
        all_results['estimated_alpha'].append(estimated_alpha)
    # else, do not provide estimate for alpha parameter
    else:
        all_results['estimated_model'].append(model_string)
        all_results['estimated_alpha'].append(np.NaN)

    all_results['duration_inference_alpha'].append(alpha_time)

    # save at every 100 alignments
    if count % 100 == 0:
        dt = pd.DataFrame.from_dict(all_results)
        dt.to_csv(results_filename)
        print('saving after', count)

dt = pd.DataFrame.from_dict(all_results)
dt.to_csv(results_filename)

