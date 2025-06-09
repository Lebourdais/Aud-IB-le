"""
Function heavily inspired by the essentia toolkit (most of the time translated from c++)
"""
from base_functions import *
from math import sqrt
def hz2bark(f):
    """
    Coming from the essentia toolkit
    """
    b = ((26.81*f)/(1960+f))-0.53
    if b<2:
        b +=0.15*(2-b)
    if b > 20.1:
        b+=0.22*(b-20.1)
    return b

def spec_hz2bark(spec,freqs):
    mapping = {ii:int(hz2bark(f)) for ii,f in enumerate(freqs)}
    inverse_mapping = {}
    for bin,bark in mapping.items():
        if bark not in inverse_mapping:
            inverse_mapping[bark] = [bin]
        else:
            inverse_mapping[bark].append(bin)
    barkspec = torch.zeros((spec.shape[0],24,spec.shape[2]))
    for ii,b in enumerate(spec): #To be batched
        # b : freq, time
        for bark_bin in inverse_mapping:
            val = spec[ii,inverse_mapping[bark_bin],:].sum(0)
            barkspec[ii,bark_bin] = val
    return barkspec

def barkCriticalBandwidth(z):
  return 52548.0 / (z*z - 52.56 * z + 690.39)

def bark2hz(b):
    """
    Coming from the essentia toolkit
    """
    if (b < 2):
        b = (b - 0.3) / 0.85
    if (b > 20.1): 
        b = (b - 4.422) / 1.22
    return 1960.0 * (b + 0.53) / (26.28 - b)

def aWeighting(f):
    # from http://www.cross-spectrum.com/audio/weighting.html
    # Weight a frequency
    return (12200**2 *(f**4)) / ((f**2 +20.6**2) *(f**2 +12200**2) * sqrt(f**2 +107.7**2) * sqrt(f**2 +737.9**2))

def plompLevelt( df):
    if (df < 0) : return 1
    if (df > 1.18) : return 1
    res =(-6.58977878 * df**5 +
        28.58224226 * df**4 +
        -47.36739986 * df**3 +
        35.70679761 * df**2 +
        -10.36526344 * df +
        1.00026609)
    if (res < 0) : return 0
    if (res > 1) : return 1
    return res

def consonance(f1,f2):
    cbwf1 = barkCriticalBandwidth(hz2bark(f1))
    cbwf2 = barkCriticalBandwidth(hz2bark(f2))
    cbw = min(cbwf1, cbwf2 )
    return plompLevelt(abs(f2-f1)/cbw)

def calcDissonance(spec,freqs):
    """
    considered loudness as energy 
    """
    out = []
    bin2Hz = {ii:freq for ii,freq in enumerate(freqs)}
    weights = torch.tensor([aWeighting(bin2Hz[f]) for f in bin2Hz]).unsqueeze(1)
    
    for b_spec in spec: #Not dealing with batch for now
        dissonance = torch.zeros(b_spec.shape[1])
        

        weigted_b_spec = b_spec * weights**2
        total_loudness = weigted_b_spec.sum(0)
        
        for t in range(b_spec.shape[1]):
            total_dissonance = 0
            if total_loudness[t]!=0:
                for f1 in range(b_spec.shape[0]):
                    peak_dissonnance = 0
                    for f2 in range(f1):
                        d = 1.0 - consonance(bin2Hz[f1],bin2Hz[f2])
                        if d > 0:
                            peak_dissonnance += d * (weigted_b_spec[f2,t]+weigted_b_spec[f1,t])/total_loudness[t]
                    partialLoudness = weigted_b_spec[f1,t]/total_loudness[t]
                    if (peak_dissonnance > partialLoudness):  peak_dissonnance = partialLoudness
                    total_dissonance+=peak_dissonnance
            dissonance[t] = total_dissonance/2
        out.append(torch.tensor(dissonance))
    return torch.stack(out)



