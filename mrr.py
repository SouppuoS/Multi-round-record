import os
import random
import argparse
import numpy as np
import soundfile as sf
from itertools import combinations

def getWavInSpkOrder(path):
    data = []
    for rt, _, file in os.walk(path):
        if len(file) == 0:
            continue
        path_a = [os.path.join(rt, fn) for fn in file]
        data.append({'spk': os.path.basename(rt), 'utt': path_a})
    return data

def generate(data, n_src, k_min, k_max, beta, R, outpth):
    
    p_mix = os.path.join(outpth, 'mix_clean')
    p_gt  = []
    os.makedirs(p_mix, exist_ok=True)
    for i in range(n_src):
        p_gt.append(os.path.join(outpth, f's{i + 1}'))
        os.makedirs(p_gt[i], exist_ok=True)
        
    for spkIds in combinations(range(len(data)), n_src):
        gt    = [np.zeros(1) for _ in range(n_src)]
        t     = 0
        fname = ''
        for spk in spkIds:
            fname += data[spk]['spk'] + '-'
        for _ in range(k_min, k_max + 1):
            for i, spk in enumerate(spkIds):
                utts   = data[spk]['utt']
                uttId  = random.randint(0, len(utts) - 1)
                wav, _ = sf.read(utts[uttId])
                # 16khz->8khz
                wav    = wav[::2]
                dB     = (random.random() * 2 - 1) * R
                wav   *= 10 ** (dB / 20.)
                
                t      = 0 if t == 0 else max(t + random.randint(-beta, beta), 0)
                t_new  = t + len(wav)
                
                gt[i] = np.pad(gt[i], (0, t_new - len(gt[i])))
                gt[i][t:t_new] = wav
                
                t = t_new
        gt = [np.pad(g, (0, t - len(g))) for g in gt]
        o  = np.zeros(gt[0].shape)
        for i in range(n_src):
            sf.write(os.path.join(p_gt[i], fname + '.wav'), gt[i], 8000, subtype='FLOAT')
            o += gt[i]
        sf.write(os.path.join(p_mix, fname + '.wav'), o, 8000, subtype='FLOAT')
    
def genWsj0MRR(path, conf):
    data = getWavInSpkOrder(path)
    generate(data, 2, conf.kmin, conf.kmax, conf.beta, 2.5, conf.o)
        

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--tr",    default=None,     type=str,   help='path of wsj0 tr data')
    parse.add_argument("--cv",    default=None,     type=str,   help='path of wsj0 cv data')
    parse.add_argument("--tt",    default=None,     type=str,   help='path of wsj0 tt data')
    parse.add_argument("--o",     default='tmp',    type=str,   help='output path')
    parse.add_argument("--kmin",  default=1,        type=int,   help='k_min')
    parse.add_argument("--kmax",  default=1,        type=int,   help='k_max')
    parse.add_argument("--beta",  default=0,        type=int,   help='shift range')
    
    conf     = parse.parse_args()
    pathData = [conf.tr, conf.cv, conf.tt]
    for p in pathData:
        if p is not None:
            genWsj0MRR(p, conf)