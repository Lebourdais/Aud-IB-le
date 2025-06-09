import matplotlib.pyplot as plt
from essentia import spec_hz2bark,calcDissonance
from base_functions import *
from matplotlib.widgets import Button, Slider
fig = plt.figure(layout="constrained")
sr = 16000
length = 3.0
step_length =0.01
win_length = 0.05
sig_displayed = 0
#sig,metadata = sig_generator(3.0,sr)
sig,metadata = load_sample_ESC50_dataset("/mnt/data/ESC-50-master")
# Create plot grid
gs0 = fig.add_gridspec(6,1)
ax_slider = fig.add_subplot(gs0[0])
sig_slider = Slider(ax=ax_slider,label="Sig displayed",valmin=0,valmax=7,valstep=1.0,handle_style={"": "^", "size":20, "facecolor":"r"})
ax_sig = fig.add_subplot(gs0[1])
gs1 = gs0[2].subgridspec(1,2)
ax_spec1 = fig.add_subplot(gs1[0])
ax_spec2 = fig.add_subplot(gs1[1])
gs2 = gs0[3].subgridspec(1,2)
ax_e1 = fig.add_subplot(gs2[0])
ax_e2 = fig.add_subplot(gs2[1])
gs3 = gs0[4].subgridspec(1,2)
ax_fft = fig.add_subplot(gs3[0])
ax_rms = fig.add_subplot(gs3[1])
gs4 = gs0[5].subgridspec(1,2)
ax_bark = fig.add_subplot(gs4[0])
ax_bark2 = fig.add_subplot(gs4[1])

# Compute everything
spec,spec_bins = spectrogram(sig,sr,db=True)
e = energy(sig,samplerate=sr,win_func=torch.signal.windows.hann)
e2 = energy(sig,samplerate=sr,win_func=torch.ones)
print(f"{dynamic_range(sig,sr)=}")
freqs_all = fft(sig,sr)
freq_bins = freqs_all[1]
freq_rep = freqs_all[0]
freq_range = (0,8000)
range_rms = (freq_range[0],freq_range[1])
rms = spectral_rms(sig,sr,range_rms)
print("RMS = ",rms)
hfcd = hF_content_descriptor(sig,sr)
spec_band0 = spectral_energy_per_band(sig,sr)
print(f"Spectral energy for 0-150Hz : {spec_band0}")
spec_band1 = spectral_energy_per_band(sig,sr,band=(150,800))
print(f"Spectral energy for 150-800Hz : {spec_band1}")
spec_band2 = spectral_energy_per_band(sig,sr,band=(800,4000))
print(f"Spectral energy for 800-4kHz : {spec_band2}")
spec_band3 = spectral_energy_per_band(sig,sr,band=(4000,20000))
print(f"Spectral energy for 4kHz-20kHz : {spec_band3}")

bark = spec_hz2bark(spec,spec_bins)
diss = calcDissonance(spec,spec_bins)


ax_sig.set_title(f"Base sig {sig_displayed} : {metadata[sig_displayed]['title']}")
plt_sig, = ax_sig.plot(metadata[sig_displayed]['t'],sig[sig_displayed])
plt_spec = librosa.display.specshow(spec[sig_displayed].numpy(),x_axis='time',y_axis='log',ax = ax_spec1)
ax_spec1.set_ylabel("Hz")
ax_spec1.set_title("Spectrogram")
plt_e1, = ax_e1.plot(e[sig_displayed])
ax_e1.set_title("energy w hann")
plt_e2, = ax_e2.plot(e2[sig_displayed])
ax_e2.set_title("energy w/o hann")
plt_fft, = ax_fft.plot(freq_bins,freq_rep[sig_displayed])
ax_fft.set_xlim(freq_range[0],freq_range[1])
ax_fft.set_title("Fourier transform")
plt_rms_base, = ax_rms.plot(freq_bins,freq_rep[sig_displayed])
plt_rms = ax_rms.axhline(rms[sig_displayed],c="k",linestyle="--")
ax_rms.set_xlim(freq_range[0],freq_range[1])
ax_rms.set_title(f"RMS : [{range_rms[0]} - {range_rms[1]}] = {rms[sig_displayed]:.2f}")
plt_hfcd,=ax_spec2.plot(hfcd[sig_displayed])
ax_spec2.set_title("High frequency content")
plt_bark=ax_bark.imshow(torch.log10(bark[sig_displayed]))
ax_bark.invert_yaxis()
ax_bark.set_title("Bark scale log-spectrogram")
plt_diss,=ax_bark2.plot(diss[sig_displayed])
ax_bark2.set_title("Dissonance")
def update(val):
    global plt_rms
    global plt_spec
    val = int(val)
    plt_sig.set_ydata(sig[val])
    ax_sig.set_title(f"Base sig {val} : {metadata[val]['title']}")
    ax_sig.autoscale()
    plt_e1.set_ydata(e[val])
    plt_e2.set_ydata(e2[val])
    ax_e1.autoscale()
    ax_e2.autoscale()
    plt_fft.set_ydata(freq_rep[val])
    ax_fft.get_autoscaley_on()
    plt_spec.remove()
    plt_spec = librosa.display.specshow(spec[val].numpy(),x_axis='time',y_axis='log',ax = ax_spec1)
    plt_rms_base.set_ydata(freq_rep[val])
    ax_rms.get_autoscaley_on()
    plt_rms.remove()
    plt_rms = ax_rms.axhline(rms[val],c='k',linestyle="--")
    #plt_rms.set_ydata(rms[val])
    ax_rms.set_title(f"RMS : [{range_rms[0]} - {range_rms[1]}] = {rms[val]:.2f}")
    plt_hfcd.set_ydata(hfcd[val])
    ax_spec2.autoscale()
    plt_bark.set_data(torch.log10(bark[val]))
    ax_bark.autoscale()
    plt_diss.set_ydata(diss[val])
    fig.canvas.draw_idle()

sig_slider.on_changed(update)
plt.show()