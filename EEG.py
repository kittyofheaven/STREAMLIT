import math
import numpy as np
import plotly.express as px
import streamlit as st

fs_eeg = 256

# Fungsi untuk normalisasi data
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def decomposition(div, signal):
    ndat = len(signal)
    
    # Koefisien gelombang Daubechies
    h = [(1 + math.sqrt(3)) / (4 * math.sqrt(2)),
         (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
         (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
         (1 - math.sqrt(3)) / (4 * math.sqrt(2))]
    
    # Koefisien HPF (High Pass Filter)
    g = [math.pow(-1, i) * h[3 - i] for i in range(4)]

    a = [0.0] * (ndat // div)
    d = [0.0] * (ndat // div)
    
    for i in range(ndat // div):
        a[i] = 0.0
        d[i] = 0.0
        for j in range(4):
            if 2 * i + j < len(signal):
                a[i] += h[j] * signal[2 * i + j]
                d[i] += g[j] * signal[2 * i + j]
                
    return a, d

# Fungsi untuk merekonstruksi sinyal dari koefisien gelombang Daubechies
def rekonstruksi(a, d):
    ndat = len(a) * 2
    
    # Koefisien rekonstruksi gelombang Daubechies
    ih = [(1 - math.sqrt(3)) / (4 * math.sqrt(2)),
          (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
          (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
          (1 + math.sqrt(3)) / (4 * math.sqrt(2))]
    
    # Koefisien HPF untuk rekonstruksi
    ig = [math.pow(-1, i) * ih[3 - i] for i in range(4)]

    signal = [0.0] * ndat
    
    for i in range(len(a)):
        for j in range(4):
            if 2 * i + j < len(signal):
                signal[2 * i + j] += ih[j] * a[i] + ig[j] * d[i]
                
    return signal
def delta_eeg(signal):
    d_power, t_power, a_power, b_power, g_power, total_power, freq, power = calculate_band(signal) 
    return d_power
def theta_eeg(signal):
    d_power, t_power, a_power, b_power, g_power, total_power, freq, power = calculate_band(signal)
    return t_power
def alpha_eeg(signal):
    d_power, t_power, a_power, b_power, g_power, total_power, freq, power = calculate_band(signal)
    return a_power
def beta_eeg(signal):
    d_power, t_power, a_power, b_power, g_power, total_power, freq, power = calculate_band(signal)
    return b_power
def gamma_eeg(signal):
    d_power, t_power, a_power, b_power, g_power, total_power, freq, power = calculate_band(signal)
    return g_power

# Fungsi untuk plot rekonstruksi sinyal
def plotRekon(signal, time, level, title):
    time_rekon = np.arange(len(signal)) / fs_eeg
    fig = px.line(x=time_rekon, y=signal, labels={'x':'Time (s)', 'y':'Amplitude'}, title=title)
    st.plotly_chart(fig)

def recta(n):
    t = np.arange(n)
    return 1*t

def fftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    results = np.empty(n, int)
    N = (n-1)//2 + 1
    p1 = np.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = np.arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val

fbands = {
    'Delta': (0.2, 3),
    'Theta': (3, 8),
    'Alpha': (8, 13),
    'Beta' : (13, 30),
    'Gamma' : (30, 50)}

def welch(data_eeg, window, win, nfft):
    data_eeg = np.asarray(data_eeg)

    fs = 256

    shift = int(window/2)

    start = np.arange(0, len(data_eeg) + 1)[0:(len(data_eeg) + 1) - window:shift]
    end = np.arange(0, len(data_eeg) + 1)[window:len(data_eeg) + 1:shift]
    if start[-1] != len(data_eeg) - window:
        start = np.append(start, len(data_eeg) - window)
        end = np.append(end, len(data_eeg))
    
    data_eegs = []
    for i in range(len(start)):
    # for i in range(len(start)):
        data_eeg_temp = data_eeg[start[i]:end[i]]
        mean_data_eeg = np.mean(data_eeg_temp)
        data_eeg_norm = data_eeg_temp - mean_data_eeg
        data_eegs.append(win * data_eeg_norm)

    scale = 1.0 / (fs)

    x = np.fft.fft(data_eegs, n=nfft)
    x = np.conjugate(x) * x
    x *= scale
    x[..., 1:] *= 2
    x = x.mean(axis=0)
    x_real = x.real

    N = (nfft-1)//2 + 1
    freq = fftfreq(nfft, 1/fs)[:N]
    power = x_real[:N]
    power /= np.max(power)

    indices = []
#     for key in tqdm(fbands.keys(), desc= 'Calculate LF, HF and Total Power'):
    for key in fbands.keys():
        indices.append(np.where((fbands[key][0] <= freq) & (freq <= fbands[key][1])))
    
    df = (freq[1] - freq[0])
    d_i, t_i, a_i, b_i, g_i = indices[0][0], indices[1][0], indices[2][0], indices[3][0], indices[4][0]
    
    d_power = np.sum(power[d_i]) * df
    t_power = np.sum(power[t_i]) * df
    a_power = np.sum(power[a_i]) * df
    b_power = np.sum(power[b_i]) * df
    g_power = np.sum(power[g_i]) * df

    total_power = np.sum((d_power, t_power, a_power, b_power, g_power))

    return d_power, t_power, a_power,b_power, g_power, total_power, freq, power

# Calculate frequency domain
def calculate_band(signal):
    window = len(signal)
    window_function = recta(window)
    d_power, t_power, a_power, b_power, g_power, total_power, freq, power = welch(signal, window, window_function, 2**13)
    return d_power, t_power, a_power, b_power, g_power, total_power, freq, power

def mpf(freq, power):
    num = 0
    denum = 0
    for i in range (len(freq)):
        num += freq[i]*power[i]
        denum += power[i]
    return num/denum #.MPF

