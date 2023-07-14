import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import math
import torch
import pickle
import numpy as np
import commons
import soundfile as sf
import argparse
import csv
import utils 
from art_Xinghua.art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models_new import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
from torch.nn import CTCLoss
from multiprocessing import Pool

def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default = 'deep_speech', help='select model as either deepspeech or ESPNet')
    parser.add_argument('--data_set', type=str, default = 'librispeech', help='select model as either librispeech or common_voice')
    parser.add_argument('--multi_speaker', type=bool, default = True, help='Whether use the multispeaker TTS model (i.e., CVAE from ICML-2021)')
    parser.add_argument('--ctext', type=str, default = 'The University', help='the conditional text for TTS model')
    parser.add_argument('--ttext', type=str, default = 'tiktok', help='the target text for attack')
    parser.add_argument('--scale_bound', type=float, default = 7, help='the search space bound for noise')
    parser.add_argument('--max_ite', type=int, default = 8000, help='the maximum optimization steps for searching')
    parser.add_argument('--sid', type=float, default = 12, help='the speaker id for multi-speaker TTS model (In Single speaker version, sid=None)')
    parser.add_argument('--ns', type=float, default = 0.3, help='the noise scale of cvae')
    parser.add_argument('--nsw', type=float, default = 0, help='the noise scale of stochastic duration predictor. (Must be set to 0)')
    parser.add_argument('--ls', type=float, default = 1, help='length scale for duration predictor')
    parser.add_argument('--seed', type=int, default = 123, help='random seed for repoduction')
    parser.add_argument('--index', type=int, default = 0, help='the index for attacked date point')
    parser.add_argument('--patience', type=int, default = 150, help='the patience for learning rate update')
    parser.add_argument('--gpu', type=int, default = 0, help='the gpu id to use')
    parser.add_argument('--init', type=float, default = 0.03, help='the learning rate initialization')
    args = parser.parse_args()
    return args

def read_csv(dataset):
    if dataset=='librispeech':
        name = 'data_sets/librispeech/librispeech_cvae_attack_data.csv'
    elif dataset=='common_voice':
        name = 'data_sets/common_voice/common_voice_cvae_attack_data.csv'
    else:
        raise ValueError('Dataset is not specified. Choose from librispeech and common_voice')
    C_Text = []
    T_Text   = []
    ind = 0
    with open(name) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter=",")
        for line in tsvreader:
            if ind==0:
                ind+=1
                continue
            C_Text.append(line[0])
            T_Text.append(line[2])
            if ind>=100: # 100 is the length of the test dataset
                break
            ind += 1
    return C_Text, T_Text

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

class CVAE_Attack_Net(nn.Module):
    def __init__(self, asr_model, net_g, sid, ctext, ttext, hps, b_size):
        super(CVAE_Attack_Net, self).__init__()
        self.asr = asr_model
        self.tts = net_g
        self.b_size  = b_size
        self.sid = sid
        self.ctxt = ctext
        self.ttxt = ttext
        self.hps = hps
        self.stn_tst = get_text(self.ctxt, self.hps)
        self.x_tst = self.stn_tst.cuda().unsqueeze(0).repeat(self.b_size, 1)
        self.x_tst_lengths = torch.LongTensor([self.stn_tst.size(0)]).cuda()
        self.ns = 0.3
        self.nsw = 0
        self.ls = 1        
        self.asr._model.train()
        ## 
        self.alpha = 1
        self.beta  = 10
        
    def forward(self, n2):
        x, m_p, logs_p, x_mask = self.tts.enc_p(self.x_tst, self.x_tst_lengths)
        self.n2 = n2
        self.n2.requires_grad = True
        noise_1 = torch.zeros(x.size(0), 2, x.size(2)).cuda()
        m_p = self.tts.get_m_p(self.x_tst, self.x_tst_lengths, sid = self.sid, noise_scale=self.ns, noise_scale_w=self.nsw, length_scale=self.ls)
        audio = self.tts.infer(self.x_tst, self.x_tst_lengths, sid = self.sid, noise_scale=self.ns, noise_scale_w=self.nsw, length_scale=self.ls, noise_1 = noise_1, noise_2 = n2)[0][:,0]
        self.audio = audio
        audio_length = audio.cpu().data.float().numpy().shape
        aud_lens = [int(audio_length[1]) for i in range(self.b_size)]
        loss, decoded_output = self.asr.compute_loss_and_decoded_output(audio, [self.ttxt], real_lengths=aud_lens)
        var_n2, mean_n2 = torch.var_mean(n2, unbiased=False)
        reg = 100*(torch.abs(torch.abs(mean_n2)-0.01)+torch.abs(var_n2-1))
        return loss+reg, decoded_output

def lr_schedule(x, U, M, L, LM, Min, Ux, Mx, LMx, Lx):
    if x <=Ux:
        lr = U - x/Ux*(U-M)
        return lr
    elif x>Ux and x<=Mx:
        lr = M - (x-Ux)/(Mx-Ux)*(M-L)
        return lr
    elif x>Mx and x<=LMx:
        lr = L - (x-Mx)/(LMx-Mx)*(L-LM)
        return lr
    elif x>LMx and x<=Lx:
        lr = LM - (x-LMx)/(Lx-LMx)*(LM-Min)
        return lr
    else:
        lr = Min
        return lr
    
def plateau_decay(patience, lr, args):
    if patience>=args.patience:
        lr =lr*0.6
        patience = 0
        if lr<=1e-4:
            lr = 1e-4
        return patience, lr
    else:
        if lr<=1e-4:
            lr = 1e-4
        return patience, lr

def parse_transcript(path):
    with open(path, 'r', encoding='utf8') as f:
        transcript = f.read().replace('\n', '')
    result = list(filter(None, [labels_map.get(x) for x in list(transcript)]))
    return transcript, result

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

def main(args):   
    model_type = args.model_type
    if model_type =='deep_speech':
        # Create a DeepSpeech estimator
        asr_model = PyTorchDeepSpeech(pretrained_model="librispeech", device_type='gpu') # load the deepspeeech model (pytorch Sarah)
        asr_model._version = 3
        asr_model._device  = 'cuda'
    elif model_type =='espnet':
        #load the espnet model
        lang = 'en'
        fs = 16000 #@param {type:"integer"}
        tag = 'Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave' 
        d = ModelDownloader()
        # It may takes a while to download and build models
        asr_model = Speech2Text(d.download_and_unpack(tag), device="cuda",minlenratio=0.0,maxlenratio=0.0,ctc_weight=0.3,beam_size=10,batch_size=0,nbest=1)
    else:
        raise ValueError('ERROR: ASR model not specified, please set \'model_type\' as either \'deepspeech\' or \'espnet\'')

    multi_speaker = args.multi_speaker
    try:
        multi_speaker
    except NameError:
        raise ValueError('multi_speaker not defined')

    if not multi_speaker:
        ## single speaker model load
        hps = utils.get_hparams_from_file("./configs/ljs_base.json")
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        net_g.cuda()
        _ = net_g.eval()
        _ = utils.load_checkpoint("./pretrained_ljs.pth", net_g, None)
        print('*** Single speaker model successfully loaded ***')
    else:
        ## multi speaker model load
        hps = utils.get_hparams_from_file("./configs/ljs_base.json")
        hps_ms = utils.get_hparams_from_file("./configs/vctk_base.json")

        net_g = SynthesizerTrn(
            len(symbols),
            hps_ms.data.filter_length // 2 + 1,
            hps_ms.train.segment_size // hps.data.hop_length,
            n_speakers=hps_ms.data.n_speakers,
            **hps_ms.model)
        net_g.cuda()
        _ = net_g.eval()
        _ = utils.load_checkpoint("./pretrained_vctk.pth", net_g, None)
        print('*** Multi speaker model successfully loaded ***')

    b_size = 1
    speaker_id = args.sid
    if multi_speaker:
        sid = torch.LongTensor([speaker_id]).cuda() # speaker identity
    else:
        sid = None
    conditional_text = args.ctext
    ctext = conditional_text.upper()
    target_text = args.ttext
    target_text = target_text.upper()
    ttext = [target_text for i in range(b_size)]
    stn_tst = get_text(conditional_text, hps)
    stn_tst = stn_tst.cuda()
    x_tst = stn_tst.unsqueeze(0)
    x_tst = x_tst.repeat(b_size,1)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

    with torch.no_grad():
        x, m_p, logs_p, x_mask = net_g.enc_p(x_tst, x_tst_lengths)
        noise_1 = 1*torch.zeros(x.size(0), 2, x.size(2)).cuda()
        m_p = net_g.get_m_p(x_tst, x_tst_lengths, sid = sid, noise_scale=args.ns, noise_scale_w=args.nsw, length_scale=args.ls)
        noise_1_dim = x.size(2)
        noise_2 = torch.randn(m_p.size(), dtype=m_p.dtype, layout=m_p.layout, device=m_p.device)
        noise_2_dim = m_p.size(2)
        audio = net_g.infer(x_tst, x_tst_lengths, sid = sid, noise_scale=args.ns, noise_scale_w=args.nsw, length_scale=args.ls, noise_1 = noise_1, noise_2 = noise_2)[0][:,0]
    print('## CVAE model works properly. Will start to do the attack ##')
    ## Combine the gradient sign and gradient magnitude method together
    my_model = CVAE_Attack_Net(asr_model, net_g, sid, ctext, target_text, hps, b_size)
    torch.manual_seed(args.seed)
    n_2 = torch.randn(m_p.size(), dtype=m_p.dtype, layout=m_p.layout, device=m_p.device, requires_grad=True)
    critenzer = CTCLoss()
    Max_Ite = args.max_ite

    

def FUN(i):
    args       = myargs()
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)
    args.data_set = 'mnist_patience_init_{}_{}'.format(args.init, args.patience)
    CText      = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Zero']
    TText      = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Zero']
    args.ctext = CText[i].upper()
    candidates = TText
    for j in range(len(candidates)):
        args.ttext = candidates[j].upper()
        print('ctext: {} ttext:{}'.format(args.ctext, args.ttext))
        main(args)
    
if __name__== "__main__":
    args       = myargs()
    args.data_set = 'mnist_patience_init_{}_{}'.format(args.init, args.patience)
    args.ctext = 'TEST'
    args.ttext = 'TEST'
    main(args)