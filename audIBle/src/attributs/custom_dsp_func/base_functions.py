import torchaudio
import torch
import librosa
import pandas as pd # Only for dataset testing
import math
def sig_generator(length,samplerate,batch_size=8):
    t = torch.linspace(0.0,length,steps=int(samplerate*length))
    freq_base = (100 / length)
    freq_range = freq_base*0.50,freq_base*1.5
    amplitude_range = 0.5,1.5
    out = []
    metadata = {i:{'t':t}for i in range(batch_size)}
    for i in range(batch_size):
        num_sum = torch.randint(low = 3,high=8,size=(1,))
        metadata[i]['num_signals'] = num_sum.item()
        metadata[i]['freqs'] = []
        metadata[i]['w'] = []
        sig_glob = None
        for _ in range(num_sum):
            A = (amplitude_range[1]-amplitude_range[0])*torch.rand(1) + amplitude_range[0]
            freq = (freq_range[1]-freq_range[0])*torch.rand(1) + freq_range[0]
            metadata[i]['freqs'].append(freq.item())
            w = 2 * torch.pi * freq
            sig = A * torch.sin(w*t)
            if sig_glob is None:
                sig_glob = sig
            else:
                sig_glob += sig
        normed_sig = (sig_glob - sig_glob.min())/(sig_glob.max()-sig_glob.min())*2 -1
        out.append(normed_sig)
        metadata[i]['title'] = f"[{', '.join([f'{x:.2f}' for x in metadata[i]['freqs']])}]"
    return torch.stack(out),metadata

def load_sample_ESC50_dataset(path,batchsize=8, target_sr:int = 16000):
    """
    load a part of ESC-50 for testing
    """
    data = pd.read_csv(f"{path}/meta/esc50.csv")
    batch = data.sample(batchsize,ignore_index=True)
    t = torch.linspace(0,5.0,steps=16000*5)
    metadata = {i:{'t':t}for i in range(batchsize)}
    full_batch = []
    for ii,r in batch.iterrows():
        audio,sr = torchaudio.load(f"{path}/audio/{r['filename']}")
        if sr != target_sr:
            rs = torchaudio.transforms.Resample(orig_freq=sr,new_freq=target_sr)
            audio = rs(audio).squeeze()
        full_batch.append(audio)
        metadata[ii]['title'] = f"{r['filename']} : {r['category']}"
    return torch.stack(full_batch),metadata
        

def loudness(signal,samplerate):
    return torchaudio.functional.loudness(signal,samplerate)

def stich(windowed_sig,base_shape,stepsize,samplerate,win_func = torch.ones):
    step = int(stepsize*samplerate)
    tmp = []
    for win_sig in windowed_sig: # not batched yet
        basesig = torch.zeros(base_shape)
        counter = torch.zeros(base_shape)
        window = win_func(win_sig.shape[-1])
        for ii,sig in enumerate(win_sig):
            basesig[step*ii:step*ii + sig.shape[0]]+=sig
            counter[step*ii:step*ii + sig.shape[0]]+=window
        tmp.append(basesig/counter)
    return torch.stack(tmp)

def fft(sig,samplerate):
    fft_val = torch.fft.fft(sig).abs() #useful values are under 100Hz
    freq = torch.tensor([ii * (samplerate/fft_val.shape[-1]) for ii in range(fft_val.shape[-1])]) # Bins are every Fs/N Hz, with N the size of fft
    return (fft_val,freq)

def windowing(sig,winsize,stepsize,samplerate,dimension = -1,win_func=torch.ones):
    win_len = int(winsize*samplerate)
    step_len = int(stepsize*samplerate)
    siglen = sig.shape[-1]
    padding = (0,win_len-step_len)
    padded_signal = torch.nn.functional.pad(sig,padding,"constant")
    segs = sig.unfold(dimension,win_len,step_len)
    window = win_func(win_len).unsqueeze(0)
    return segs * window

def energy(signal,samplerate,winsize = 0.05,stepsize=0.01,win_func=torch.ones):
    
    signal_window = windowing(signal,winsize,stepsize,samplerate,win_func=win_func) # batch,n_win,s_win
    energy = signal_window.abs().pow(2).sum(dim=-1)
    return energy

def energy_win(window):
    energy = window.abs().pow(2).sum(dim=-1)
    return energy

def windowed_acf(signal,samplerate,window = 1,step = 0.01):
    # windowed autocorrelation function
    windowed_sig = windowing(signal,winsize=window,stepsize=step,samplerate=samplerate)
    energy = energy_win(windowed_sig)
    starting_point_b = [energy[b].argmax() for b in range(signal.shape[0])]
    #starting_point_b = torch.randint(low=3,high=windowed_sig.shape[1]-5,size=(windowed_sig.shape[0],))
    #print(f"{signal.shape=}")
    steps = torch.arange(windowed_sig.shape[1])
    x_glob = steps.float() * step + window / 2
    
    x_acc = []
    #print(f"{x_glob=}")
    acc = []
    peaks_glob = []
    for windowb, starting_point in zip(windowed_sig,starting_point_b):
        starting_point_s = starting_point.item()
        #print(f"{windowb.shape=}")
        sigs= windowb
        kernel = windowb[starting_point_s,:].unsqueeze(0)
        #print(f"{sigs.shape=}, {kernel.shape=}")
        corr_function = (sigs * kernel).sum(dim=1)
        x = [(x_glob[y]) - (x_glob[starting_point_s]) for y in range(windowb.shape[0])]
        #print(f"{corr_function.shape=}")
        x_acc.append(x)
        acc.append(corr_function/max(corr_function))
        peaks = torch.topk(corr_function/max(corr_function),10).values[1:]
        peaks_glob.append(torch.tensor(peaks))
        #print(f"{peaks=}")
    return torch.stack(acc),x_acc,torch.stack(peaks_glob)

def multi_level_acf(signal,samplerate,space=(0.1,1,100)):
    x = torch.linspace(start=space[0],end=space[1],steps=space[2])
    y = []
    
    for win in x:
        acf,x_acc,peaks = windowed_acf(signal,samplerate,window=win)
        mean_acf = peaks.mean(dim=1)
        y.append(mean_acf)
    #print(f"{torch.stack(y).T.shape=},{x.repeat(signal.shape[0],0).shape=}")
    return torch.stack(y).T,x

def dynamic_range(signal,samplerate):
    e = energy(signal=signal,samplerate=samplerate)
    return 20 * torch.log10(e.max(dim=1).values/e.min(dim=1).values)

def spectral_rms(signal,samplerate,freqrange = (0,8000)):
    freq_rep,freq_bins = fft(signal,samplerate)
    freq_mask = (freqrange[0] < freq_bins) & (freq_bins < freqrange[1])
    out = []
    for batch_raw in freq_rep:
        batch = batch_raw[freq_mask]
        rms = torch.sqrt((batch.pow(2)).sum()/batch.shape[0])
        out.append(rms)
    return torch.tensor(out)

def spectral_energy_per_band(signal,samplerate,band=(20,150)):
    """
    [20Hz, 150Hz], [150Hz, 800Hz], [800Hz, 4kHz], and [4kHz, 20kHz]
    """
    freq_rep,freq_bins = spectrogram(signal,samplerate,power=1.0) # energy not power
    freq_bins = torch.tensor(freq_bins)
    freq_mask = ((band[0] < freq_bins) & (freq_bins < band[1])).unsqueeze(0).repeat(freq_rep.shape[0],1)
    energy = freq_rep[freq_mask].abs().pow(2).sum()
    return energy

def spectrogram(signal,samplerate,n_fft=400,db=False,power = 2.0):
    n_bins = n_fft
    spec_func = torchaudio.transforms.Spectrogram(n_fft=n_bins,power=power)
    spec = spec_func(signal)
    freq_bins = [(i * samplerate/n_bins)for i in range(spec.shape[1])]
    if db:
        spec = 10 * torch.log10(spec)
    return spec,freq_bins

def hF_content_descriptor(signal,samplerate):
    spec,freqs = spectrogram(signal,samplerate)
    weights = torch.arange(len(freqs))
    weighted_spec = spec*weights.unsqueeze(1)
    hfcd = weighted_spec.sum(1)
    return hfcd
# loudness OK
# dynamic range OK
# spectral rms OK
# HF content descriptor OK
# spectral energy per band OK
#dissonance OK
#pitch salience
#bpm
#danceability
#key
#Chord change rate
