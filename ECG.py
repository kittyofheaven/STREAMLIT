import pandas as pd
import numpy as np
import math
import seaborn as sns
import biosppy
import pyhrv.nonlinear as nl 
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.signal import welch

def dirac(x):
    return 1 if x==0 else 0

def get_array_coefficient(scale:int):
    power_two = np.round(2**scale)
    power_two_min = np.round(2**(scale-1))
    array_coef = np.arange(-(power_two+power_two_min-2), power_two_min)
    array_coef1 = np.arange((1-power_two_min), (power_two+power_two_min-2) + 1)
    return array_coef, array_coef1

def dwt_filter_bank():
    #Filter Coeff
    k1, skip = get_array_coefficient(scale = 1)
    k2, skip = get_array_coefficient(scale = 2)
    k3, skip = get_array_coefficient(scale = 3)

    #Impulse Response
    qj1 = np.zeros((len(k1)))
    qj2 = np.zeros((len(k2)))
    qj3 = np.zeros((len(k3)))

    for index, x in enumerate(k1):
        qj1[index] = -2*(dirac(x)-dirac(x+1))
    
    for index, x in enumerate(k2):
        qj2[index] = -1/4*(dirac(x-1) + 3*dirac(x) + 2*dirac(x+1) - 2*dirac(x+2) - 3*dirac(x+3) - dirac(x+4))
     
    for index, x in enumerate(k3):
        # qj3[index] = -1/32*(dirac(x+3) + 3*dirac(x+2) + 6*dirac(x+1) + 10*dirac(x) + 11*dirac(x-1) + 9*dirac(x-2) + 4*dirac(x-3) - 4*dirac(x-4) - 9*dirac(x-5) - 11*dirac(x-6) - 10*dirac(x-7) - 6*dirac(x-8) - 3*dirac(x-9) - dirac(x-10))
        qj3[index] = -1/32*(dirac(x-3) + 3*dirac(x-2) + 6*dirac(x-1) + 10*dirac(x) + 11*dirac(x+1) + 9*dirac(x+2) + 4*dirac(x+3) - 4*dirac(x+4) - 9*dirac(x+5) - 11*dirac(x+6) - 10*dirac(x+7) - 6*dirac(x+8) - 3*dirac(x+9) - dirac(x+10))
    return k1, k2, k3, qj1, qj2, qj3

def delay(scale):
  return round((2**(scale-1))) - 1

def get_index(data, index):
    if index < 0:
        return 0
    if index > data.size - 1:
        return 0
    return data[index]

def decomposition(data_ecg):
    signal_ecg = np.array(data_ecg)  # Ubah dari signal_ecg ke data_ecg
    n_data = signal_ecg.shape[0]
    w2fb1 = np.zeros(n_data + delay(1))
    w2fb2 = np.zeros(n_data + delay(2))
    w2fb3 = np.zeros(n_data + delay(3))
    k1, k2, k3, qj1, qj2, qj3 = dwt_filter_bank()
    k_temp_1, _ = get_array_coefficient(scale=1)
    k_temp_2, _ = get_array_coefficient(scale=2)
    k_temp_3, _ = get_array_coefficient(scale=3)

    for n in range(n_data - 1):
        w2fb1[n + delay(1)] = 0 
        w2fb2[n + delay(2)] = 0
        w2fb3[n + delay(3)] = 0
       
        for x1 in k_temp_1:
            w2fb1[n + delay(1)] += qj1[x1] * get_index(data_ecg, n - x1)
        for x2 in (k_temp_2):
            w2fb2[n + delay(2)] += qj2[x2] * get_index(data_ecg, n - x2)
        for x3 in (k_temp_3):
            w2fb3[n + delay(3)] += qj3[x3] * get_index(data_ecg, n - x3)
    
    return w2fb1, w2fb2, w2fb3

def detectpeak(gradien1, gradien2, gradien3, n4):
    hasil1 = np.zeros(n4 + 1, dtype=int)
    hasil2 = np.zeros(n4 + 1, dtype=int)
    hasil3 = np.zeros(n4 + 1, dtype=int)
    sisa1 = np.zeros(n4 + 1, dtype=int)
    
    for n in range(n4 + 1):
        if gradien1[n] > 1.2:
            hasil1[n] = 1
            for i in range(max(0, n - 10), min(n4, n + 10) + 1):
                sisa1[i] = 1
        if gradien2[n] > 0.35:
            hasil2[n] = 1
        if gradien3[n] > 0.2:
            hasil3[n] = 1
    
    return hasil1, hasil2, hasil3, sisa1

# Plot QRS Detection
def plot_qrs_detection(time_ecg, signal_ecg, hasil_qrs):
    
    fig = px.line(x=time_ecg[:1000], y=signal_ecg[:1000], title='QRS Detection',
                  labels={'x': 'Elapsed Time (s)', 'y': 'Amplitude (V)'})
    
    fig.add_scatter(x=time_ecg[:1000], y=hasil_qrs[:1000], mode='lines', name='R Peak Detected', 
                    line=dict(color='yellow', width=2), showlegend=False)
    
    st.plotly_chart(fig)

# Function to Calculate Peak-to-Peak and QRS Detection
def calculate_peak_to_peak_and_qrs_detection(thrqrs, fs):
    ptp = 0
    waktu = np.zeros(np.size(thrqrs))
    selisih = np.zeros(np.size(thrqrs))

    for n in range(np.size(thrqrs) - 1):
        if thrqrs[n] < thrqrs[n + 1]:
            waktu[ptp] = n / fs
            if ptp > 0:
                selisih[ptp] = waktu[ptp] - waktu[ptp - 1]
            ptp += 1

    ptp -= 1

    j = 0
    peak = np.zeros(np.size(thrqrs))
    for n in range(np.size(thrqrs)-1):
        if thrqrs[n] == 1 and thrqrs[n-1] == 0:
            peak[j] = n
            j += 1

    return ptp, j

# Time Domain Analysis
def calculate_time_domain_features(intervals):
    # RMSSD calculation
    rr_diff = np.diff(intervals)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    
    # pNN50 calculation
    nn50 = sum(abs(rr_diff) > 0.05)
    pnn50 = (nn50 / len(intervals)) * 100
    
    # SDNN calculation
    sdnn = np.std(intervals)
    
    # SDSD calculation
    # rr_diff_sq = np.diff(rr_diff)
    # sdsd = np.sqrt(np.mean(rr_diff_sq ** 2))
    
    return rmssd, pnn50, sdnn
 # Frequency Domain Analysis
def hanning(input_segment):
    n_seg = len(input_segment)
    window = np.zeros(n_seg)
    for i in range(n_seg):
        window[i] = input_segment[i] * (0.5-0.5*np.cos(2*np.pi*i/n_seg))
    return window

def hanning_new(n):
    t = np.arange(n)
    return (0.5-0.5*np.cos((2*np.pi*t)/(n-1)))

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
    'vlf': (0.000, 0.04),
    'lf': (0.04, 0.15),
    'hf': (0.15, 0.4)
}

def rri_format(rr):
    if np.max(rr) < 10:
        rr_ = np.asarray(rr, dtype='float64')
        rr_ *= 1000
        return rr_
    return rr

def welch(rri, window, win, nfft):
    rri = np.asarray(rri)
    rri = rri_format(rri)

    fs = 2
    t = np.cumsum(rri)
    t -= t[0]
    f_interpol = interpolate.interp1d(t, rri, 'cubic')
    t_interpol = np.arange(t[0], t[-1], 1000./fs)
    rri = f_interpol(t_interpol)
    rri = rri - np.mean(rri)

    window = min(1198, (len(rri)))

    shift = int(window/2)

    start = np.arange(0, len(rri) + 1)[0:(len(rri) + 1) - window:shift]
    end = np.arange(0, len(rri) + 1)[window:len(rri) + 1:shift]
    if start[-1] != len(rri) - window:
        start = np.append(start, len(rri) - window)
        end = np.append(end, len(rri))
    
    rris = []
    for i in range(len(start)):
    # for i in range(len(start)):
        rri_temp = rri[start[i]:end[i]]
        mean_rri = np.mean(rri_temp)
        rri_norm = rri_temp - mean_rri

        n = len(rri_norm)
        t = np.arange(n)
        win = 0.5-0.5*np.cos((2*np.pi*t)/(n-1))
        rris.append(win * rri_norm)

    scale = 1.0 / (fs * (win**2).sum())

    x = np.fft.fft(rris, n=nfft)
    x = np.conjugate(x) * x
    x *= scale
    x[..., 1:] *= 2
    x = x.mean(axis=0)
    x_real = x.real

    N = (nfft-1)//2 + 1
    freq = fftfreq(nfft, 1/fs)[:N]
    power = x_real[:N]

    indices = []
    for key in fbands.keys():
    # for key in fbands.keys():
        indices.append(np.where((fbands[key][0] <= freq) & (freq <= fbands[key][1])))
    
    df = (freq[1] - freq[0])
    vlf_i, lf_i, hf_i = indices[0][0], indices[1][0], indices[2][0]
    
    vlf_power = np.sum(power[vlf_i]) * df
    lf_power = np.sum(power[lf_i]) * df
    hf_power = np.sum(power[hf_i]) * df

    total_power = np.sum((vlf_power, lf_power, hf_power))

    return vlf_power, lf_power, hf_power, total_power, freq, power


# Non Linear Analysis (SD1, SD2) using Poincare plot
def non_linear(intervals):
    results = nl.poincare(nni=intervals) 
    sd1_value = results['sd1']
    sd2_value = results['sd2']
    return sd1_value, sd2_value

# Main ECG Function
def ecg(signal_ecg, fs_ecg=125):
    w2fb1, w2fb2, w2fb3 = decomposition(signal_ecg)
    ndata = signal_ecg.shape[0]
    time_ecg = np.arange(ndata) / fs_ecg
    
    gradien1 = np.diff(w2fb1, prepend=0)
    gradien2 = np.diff(w2fb2, prepend=0)
    gradien3 = np.diff(w2fb3, prepend=0)

    n4 = ndata - 1
    hasil1, hasil2, hasil3, sisa1 = detectpeak(gradien1, gradien2, gradien3, n4)
    
    hasil_qrs = np.zeros(ndata)
    for n in range(ndata):
        if hasil1[n] == 1 or hasil2[n] == 1 or hasil3[n] == 1 or sisa1[n] == 1:
            hasil_qrs[n] = 1

    peakmaksQRS = [n for n in range(1, ndata) if hasil_qrs[n] == 1 and hasil_qrs[n-1] == 0]
    peak_times = np.array(peakmaksQRS) / fs_ecg
    intervals = np.diff(peak_times)

    filtered_intervals = intervals[(intervals >= 0.5) & (intervals <= 1.0)]

    if len(filtered_intervals) == 0:
        print("Warning: No valid RR intervals after filtering.")
        return {
            "intervals": intervals,
            "mean_interval": np.nan,
            "bpm": np.nan,
            "rmssd": np.nan,
            "pnn50": np.nan,
            "sdnn": np.nan,
            "vlf": np.nan,
            "lf": np.nan,
            "hf": np.nan,
            "total_power": np.nan,
            "lf_hf_ratio": np.nan,
            "sd1": np.nan,
            "sd2": np.nan,
            "ptp": np.nan,
            "qrs_detected": np.nan,
        }

    bpm = 60 / np.mean(filtered_intervals)
    rmssd, pnn50, sdnn = calculate_time_domain_features(intervals)
    sd1_value, sd2_value = non_linear(intervals)
    
    window = min(1198, (len(filtered_intervals)))
    win = hanning_new(window)
    vlf_power, lf_power, hf_power, total_power, freq, power = welch(filtered_intervals, window, win, nfft=1024)

    ptp, qrs_detected = calculate_peak_to_peak_and_qrs_detection(hasil_qrs, fs_ecg)

    plot_qrs_detection(time_ecg, signal_ecg, hasil_qrs)

    print("HRV FEATURES")
    print("Time Domain")
    print(f"Mean RR : {np.mean(filtered_intervals):.4f}")
    print(f"Mean HR : {bpm:.4f}")
    print(f"RMSSD : {rmssd:.4f}")
    print(f"SDNN : {sdnn:.4f}")
    print(f"pNN50 : {pnn50:.4f}\n")

    print("Frequency Domain")
    print(f"VLF Power : {vlf_power:.6f}")
    print(f"LF Power : {lf_power:.6f}")
    print(f"HF Power : {hf_power:.6f}")
    print(f"LF Norm : {100 * lf_power / (lf_power + hf_power):.4f}")
    print(f"HF Norm : {100 * hf_power / (lf_power + hf_power):.4f}")
    print(f"LF/HF : {lf_power / hf_power:.4f}")
    print(f"Total Power : {total_power:.4f}\n")

    print("Non-Linear (Poincare Plot)")
    print(f"SD1 : {sd1_value:.4f}")
    print(f"SD2 : {sd2_value:.4f}\n")

    print(f"Peak to Peak Count : {ptp}")
    print(f"Detected RRI Count: {len(intervals)}")
    print(f"QRS Detected : {qrs_detected}")
    
    print(f"{np.mean(filtered_intervals):.4f} {bpm:.4f} {rmssd:.4f} {sdnn:.4f} {pnn50:.4f} {lf_power / hf_power:.4f} {total_power:.4f} {sd1_value:.4f} {sd2_value:.4f}")

    features = {
        "intervals": intervals,
        "mean_interval": np.mean(filtered_intervals),
        "bpm": bpm,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "sdnn": sdnn,
        "vlf": vlf_power,
        "lf": lf_power,
        "hf": hf_power,
        "sd1": sd1_value,
        "sd2": sd2_value,
        "ptp": ptp,
        "qrs_detected": qrs_detected,
    }

    return features

def plot_welch_periodogram(freq, power, vlf_power, lf_power, hf_power):
    fig = go.Figure()

    # # Plot total power
    # fig.add_trace(go.Scatter(x=freq, y=power, mode='lines', name='Total Power', line=dict(width=0.25)))

    # Highlight VLF band
    fig.add_trace(go.Scatter(
        x=[f for f in freq if f > 0.00 and f <= 0.04],
        y=[p for f, p in zip(freq, power) if f > 0.00 and f <= 0.04],
        mode='lines',
        fill='tozeroy',
        name='VLF',
        line=dict(color='red')
    ))

    # Highlight LF band
    fig.add_trace(go.Scatter(
        x=[f for f in freq if f > 0.04 and f <= 0.15],
        y=[p for f, p in zip(freq, power) if f > 0.04 and f <= 0.15],
        mode='lines',
        fill='tozeroy',
        name='LF',
        line=dict(color='green')
    ))

    # Highlight HF band
    fig.add_trace(go.Scatter(
        x=[f for f in freq if f > 0.15 and f <= 0.4],
        y=[p for f, p in zip(freq, power) if f > 0.15 and f <= 0.4],
        mode='lines',
        fill='tozeroy',
        name='HF',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='Welch Periodogram',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power',
        legend=dict(x=0.7, y=0.95),
        showlegend=True,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(showgrid=True),
        width=2000,  # Set figure width
        height=400   # Set figure height
    )

    return fig

def plot_poincare(intervals):
    sd1_value, sd2_value = non_linear(intervals)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(intervals[:-1], intervals[1:], color='blue', alpha=0.5)
    ax.set_xlabel('NN intervals (ms)')
    ax.set_ylabel('NN intervals (ms)')
    ax.set_title('Poincare Plot')
    
    # mean_interval = np.mean(intervals)

    # # Plot sd1 line
    # sd1_line_x = [mean_interval - sd1_value / np.sqrt(2), mean_interval + sd1_value / np.sqrt(2)]
    # sd1_line_y = [mean_interval + sd1_value / np.sqrt(2), mean_interval - sd1_value / np.sqrt(2)]
    # ax.plot(sd1_line_x, sd1_line_y, color='red', linestyle='--', label='SD1')

    # # Plot sd2 line
    # sd2_line_x = [mean_interval - sd2_value / np.sqrt(2), mean_interval + sd2_value / np.sqrt(2)]
    # sd2_line_y = [mean_interval - sd2_value / np.sqrt(2), mean_interval + sd2_value / np.sqrt(2)]
    # ax.plot(sd2_line_x, sd2_line_y, color='green', linestyle='--', label='SD2')

    # circle = plt.Circle((np.mean(intervals), np.mean(intervals)), sd1_value, color='red', fill=False)
    # ax.add_artist(circle)
    ax.grid(True)
    st.pyplot(fig)
