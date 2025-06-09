import torchaudio
import torch

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
    return torch.stack(out),metadata
    
        

def loudness(signal,samplerate):
    return torchaudio.functionals.loudness(signal,samplerate)

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

def dynamic_range(signal,samplerate):
    e = energy(signal=signal,samplerate=samplerate)
    return 20 * torch.log(e.max(dim=1).values/e.min(dim=1).values)

def spectral_rms(signal,samplerate,freqrange = (0,8000)):
    freq_rep,freq_bins = fft(signal,samplerate)
    freq_mask = (freqrange[0] < freq_bins) & (freq_bins < freqrange[1])
    out = []
    for batch_raw in freq_rep:
        batch = batch_raw[freq_mask]
        rms = torch.sqrt((batch.pow(2)).sum()/batch.shape[0])
        out.append(rms)
    return torch.tensor(out)
def spectrogram(signal,samplerate,n_fft=128,win_len = 128):
    n_bins = n_fft
    spec_func = torchaudio.transforms.Spectrogram(n_fft=n_bins,win_length=win_len)
    spec = spec_func(signal)
    freq_bins = [(i * samplerate/n_bins)for i in range(spec.shape[1])]
    return spec,freq_bins
def HF_content_descriptor(signal,samplerate):
    spec,freqs = spectrogram(signal,samplerate)
    weights = torch.arange(freqs.shape[0])
    weighted_spec = spec*weights
    hfcd = weighted_spec.sum(1)
    print(hfcd.shape)
    return hfcd
# loudness OK
# dynamic range OK
# spectral rms OK
# HF content descriptor
# spectral energy per band
#dissonance
#pitch salience
#bpm
#danceability
#key
#Chord change rate
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(layout="constrained")
    sr = 16000
    length = 3.0
    step_length =0.01
    win_length = 0.05
    sig_displayed = 0
    sig,metadata = sig_generator(3.0,sr)
    gs0 = fig.add_gridspec(4,1)
    ax_sig = fig.add_subplot(gs0[0])
    ax_sig.set_title(f"Base sig {sig_displayed}")
    ax_sig.plot(metadata[sig_displayed]['t'],sig[sig_displayed])
    gs1 = gs0[1].subgridspec(1,2)
    ax_spec1 = fig.add_subplot(gs1[0])
    ax_spec2 = fig.add_subplot(gs1[1])

    spec,spec_bins = spectrogram(sig,sr,n_fft=128,win_len=50)
    ax_spec1.imshow(spec[sig_displayed])
    ax_spec1.set_yticklabels(spec_bins)
    ax_spec1.set_title("Spectrogram")
    e = energy(sig,samplerate=sr,win_func=torch.signal.windows.hann)
    e2 = energy(sig,samplerate=sr,win_func=torch.ones)
    gs2 = gs0[2].subgridspec(1,2)
    ax_e1 = fig.add_subplot(gs2[0])
    ax_e2 = fig.add_subplot(gs2[1])
    
    ax_e1.plot(e[sig_displayed])
    ax_e1.set_title("energy w hann")
    ax_e2.plot(e2[sig_displayed])
    ax_e2.set_title("energy w/o hann")
    print(f"{e.shape=}")
    print(f"{dynamic_range(sig,sr)=}")
    gs3 = gs0[3].subgridspec(1,2)
    ax_fft = fig.add_subplot(gs3[0])
    ax_rms = fig.add_subplot(gs3[1])
    freqs_all = fft(sig,sr)
    freq_bins = freqs_all[1]
    freq_rep = freqs_all[0][sig_displayed]
    ax_fft.plot(freq_bins,freq_rep)
    ax_fft.set_xlim(0,50)
    ax_fft.set_title("Fourier transform")
    range_rms = (15,50)
    rms = spectral_rms(sig,sr,range_rms)
    ax_rms.plot(freq_bins,freq_rep)
    ax_rms.axhline(rms[sig_displayed],c="k",linestyle="--")
    ax_rms.set_xlim(0,50)
    ax_rms.set_title(f"RMS : [{range_rms[0]} - {range_rms[1]}] = {rms[sig_displayed]:.2f}")
    print("RMS = ",rms)

    plt.show()