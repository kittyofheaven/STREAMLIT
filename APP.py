import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import h5py
import tensorflow as tf 
import joblib
# Disable TensorFlow v2 behavior
# tf.compat.v1.disable_v2_behavior() 
from PIL import Image
from ECG import ecg, calculate_time_domain_features, welch, non_linear, plot_welch_periodogram, plot_poincare, hanning_new
from EEG import normalize, plotRekon, decomposition, rekonstruksi, calculate_band, mpf, delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg

def main():
    st.sidebar.title("Navigation")
    menu = ["HOME", "ECG", "EEG","ANN", "ABOUT"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "HOME":
    
        #image_path = os.path.join(r"C:\Users\RARA\OneDrive - Institut Teknologi Sepuluh Nopember\Dokumen\#1 TUGAS AKHIR RARA\02. TUGAS AKHIR\4. HASIL\4. STREAMLIT\STREAMLIT BISA (PALING BENAR)\its-bme.png")
        image_path = os.path.join("its-bme.png")
        image = Image.open(image_path)
        st.image(image, width=200)
        st.header("EB184803 | TUGAS AKHIR ")
        st.subheader("JUDUL : INTEGRASI SINYAL ECG DAN EEG SEBAGAI BIOFEEDBACK SERTA BIOMARKER UNTUK ASESMEN STRES")
        # st.subheader("\n")
        # st.subheader("YASMIN FAKHIRA ICHSAN | NRP.5023201032")
        st.subheader("\n")
        
        st.subheader("Uploading Files")
        files = st.file_uploader("Choose a file for processing and features extraction", type=["txt", "csv"])
        
        if files is not None:
            file_details = {"filename": files.name, "filetype": files.type, "filesize": files.size}
            st.write(file_details)

            data = pd.read_csv(files, names=['ECG & EEG'])
            data_np = np.array(data)

            if "y" not in st.session_state:
                st.session_state.y = []
            if "fp1" not in st.session_state:
                st.session_state.fp1 = []
            if "f3" not in st.session_state:
                st.session_state.f3 = []

            no_decimal = []
            large_exponent = []
            for row in data_np:
                try:
                    initial = row[0][0]
                    value_str = row[0][1:]
                            
                    # Check for no decimal point
                    if '.' not in value_str:
                        no_decimal.append(row[0])
                        continue  # Skip this row

                    # Check for large exponent
                    if 'e+' in value_str and int(value_str.split('e+')[1]) > 2:
                        large_exponent.append(row[0])
                        continue  # Skip this row
                    
                    value = float(value_str)

                    if initial == 'a':
                        st.session_state.y.append(value)
                    elif initial == 'b':
                        st.session_state.fp1.append(value)
                    elif initial == 'c':
                        st.session_state.f3.append(value)
                except ValueError:
                    pass
                    st.write("Error on line:", row)
            
            ndata_ecg = len(st.session_state.y)
            ndata_fp1 = len(st.session_state.fp1)
            ndata_f3 = len(st.session_state.f3)
            
            print("This is st session y", len(st.session_state.y))
            print("This is st session fp1", len(st.session_state.fp1))
            print("This is st session f3", len(st.session_state.f3))

            st.header("Data Exploration")
            
            st.subheader('ECG Data')
            st.session_state.data_ecg = pd.DataFrame()
            st.session_state.data_ecg['Elapsed Time'] = (np.arange(ndata_ecg)) * (1 / 250)
            st.session_state.data_ecg['Data ECG'] = st.session_state.y
            st.dataframe(st.session_state.data_ecg.head())
            
            st.subheader('EEG Fp1 Data')
            st.session_state.data_eeg_fp1 = pd.DataFrame()
            st.session_state.data_eeg_fp1["Elapsed Time"] = (np.arange(ndata_fp1)) * (1 / 512)
            st.session_state.data_eeg_fp1['EEG Fp1'] = st.session_state.fp1
            
            mean_eeg_fp1 = np.mean(st.session_state.data_eeg_fp1['EEG Fp1'])
            signal_eeg_fp1 = st.session_state.data_eeg_fp1['EEG Fp1'] - mean_eeg_fp1
            st.session_state.data_eeg_fp1['EEG Fp1'] = normalize(signal_eeg_fp1)
            
            st.dataframe(st.session_state.data_eeg_fp1.head())
            
            st.subheader('EEG F3 Data')
            st.session_state.data_eeg_f3 = pd.DataFrame()
            st.session_state.data_eeg_f3["Elapsed Time"] = (np.arange(ndata_f3)) * (1 / 512)
            st.session_state.data_eeg_f3['EEG F3'] = st.session_state.f3
            
            mean_eeg_f3 = np.mean(st.session_state.data_eeg_f3['EEG F3'])
            signal_eeg_f3 = st.session_state.data_eeg_f3['EEG F3'] - mean_eeg_f3
            st.session_state.data_eeg_f3['EEG F3'] = normalize(signal_eeg_f3)
            
            st.dataframe(st.session_state.data_eeg_f3.head())

    elif choice == "ECG":
        st.header("Features Extraction ECG")
        st.subheader('Plot ECG Signals')

        if "data_ecg" in st.session_state and not st.session_state.data_ecg.empty:
            fig = px.line(x=st.session_state.data_ecg["Elapsed Time"], y=st.session_state.data_ecg['Data ECG'],
                          labels={'x':'Elapsed time(s)', 'y':'Amplitude(mV)'}, title='ECG Signal')
            st.plotly_chart(fig)

        if "data_ecg" in st.session_state and not st.session_state.data_ecg.empty:
            fig = px.line(x=st.session_state.data_ecg["Elapsed Time"][:1000], y=st.session_state.data_ecg['Data ECG'][:1000],
                          labels={'x':'Elapsed time(s)', 'y':'Amplitude(mV)'}, title='ECG Signal')
            st.plotly_chart(fig)

        if "y" in st.session_state and "qrs_result" not in st.session_state:
            fs_ecg = 125
            ecg_raw = np.array(st.session_state.data_ecg['Data ECG']) / 100
            mean_ecg = np.mean(ecg_raw)
            signal_ecg = ecg_raw - mean_ecg

            ecg_features = ecg(signal_ecg)
            
            st.session_state.temp_rr = ecg_features["intervals"]
            st.session_state.qrs_result = ecg_features["qrs_detected"]

        if "temp_rr" in st.session_state:
            mean_rr = ecg_features["mean_interval"]
            mean_hr = ecg_features["bpm"]
            rmssd, pnn50, sdnn = calculate_time_domain_features(st.session_state.temp_rr)

            window = min(1198, (len(st.session_state.temp_rr)))
            win = hanning_new(window)
            vlf_power, lf_power, hf_power, total_power, freq, power = welch(st.session_state.temp_rr, window, win, nfft=1024)
            sd1_value, sd2_value = non_linear(st.session_state.temp_rr)
        
            st.session_state.mean_rr = mean_rr
            st.session_state.mean_hr = mean_hr
            st.session_state.rmssd = rmssd
            st.session_state.pnn50 = pnn50
            st.session_state.sdnn = sdnn
            st.session_state.lf_power = lf_power
            st.session_state.hf_power = hf_power
            st.session_state.total_power_ecg = total_power
            st.session_state.sd1 = sd1_value
            st.session_state.sd2 = sd2_value

            time_domain_data = {
                "Metric": ["Mean RR Interval (ms)", "Mean Heart Rate (bpm)", "RMSSD", "SDNN", "pNN50"],
                "Value": [f"{mean_rr:.4f}", f"{mean_hr:.4f}", f"{rmssd:.4f}", f"{sdnn:.4f}", f"{pnn50:.4f}"]
            }
            df_time_domain = pd.DataFrame(time_domain_data)

            freq_domain_data = {
                "Metric": ["LF/HF Ratio", "Total Power"], #"VLF Power", "LF Power", "HF Power", 
                "Value": [f"{lf_power / hf_power:.8f}", f"{total_power:.6f}"] #f"{vlf_power}", f"{lf_power}", f"{hf_power}",
            }
            df_freq_domain = pd.DataFrame(freq_domain_data)

            non_linear_data = {
                "Metric": ["SD1", "SD2"],
                "Value": [f"{sd1_value:.4f}", f"{sd2_value:.4f}"]
            }
            df_non_linear = pd.DataFrame(non_linear_data)

            st.subheader("Time Domain Analysis")
            st.table(df_time_domain)

            st.subheader("Frequency Domain Analysis")
            welch_fig = plot_welch_periodogram(freq, power, vlf_power, lf_power, hf_power)
            st.plotly_chart(welch_fig)
            st.table(df_freq_domain)

            st.subheader("Non Linear Analysis")
            plot_poincare(st.session_state.temp_rr)
            st.table(df_non_linear)

    if choice == "EEG":
        submenu = st.sidebar.selectbox("SubMenu", ["Fp1", "F3"])
        st.header(f"EEG Analysis - {submenu}")

        if submenu == "Fp1":
            st.subheader("EEG Fp1 Data")
            if "data_eeg_fp1" in st.session_state and not st.session_state.data_eeg_fp1.empty:
                fig = px.line(x=st.session_state.data_eeg_fp1["Elapsed Time"],
                            y=normalize(st.session_state.data_eeg_fp1['EEG Fp1']),
                            labels={'x': 'Elapsed time(s)', 'y': 'Magnitude'},
                            title='EEG Fp1 Signal')
                st.plotly_chart(fig)

                selected_level = 8
                a_fp1 = [None] * selected_level
                d_fp1 = [None] * selected_level

                signal = normalize(st.session_state.data_eeg_fp1['EEG Fp1'])
                for level in range(selected_level):
                    div = 2
                    a_fp1[level], d_fp1[level] = decomposition(div, signal)
                    signal = normalize(a_fp1[level])  

                fig = px.line(x=range(len(d_fp1[4])),
                            y=d_fp1[4],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Detail Coefficients Level 5, Sinyal Gamma')
                st.plotly_chart(fig)

                fig = px.line(x=range(len(d_fp1[5])),
                            y=d_fp1[5],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Detail Coefficients Level 6, Sinyal Theta')
                st.plotly_chart(fig)

                fig = px.line(x=range(len(d_fp1[6])),
                            y=d_fp1[6],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Detail Coefficients Level 7, Sinyal Alpha')
                st.plotly_chart(fig)

                fig = px.line(x=range(len(d_fp1[7])),
                            y=d_fp1[7],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Detail Coefficients Level 8, Sinyal Beta')
                st.plotly_chart(fig)

                fig = px.line(x=range(len(a_fp1[7])),
                            y=a_fp1[7],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Approximation Coefficients Level 8, Sinyal Delta')
                st.plotly_chart(fig)
                

                delta_band_fp1 = delta_eeg(a_fp1[7])
                theta_band_fp1 = theta_eeg(d_fp1[7])
                alpha_band_fp1 = alpha_eeg(d_fp1[6])
                beta_band_fp1 = beta_eeg(d_fp1[5])
                gamma_band_fp1 = gamma_eeg(d_fp1[4])

                dpower_fp1, tpower_fp1, apower_fp1, bpower_fp1, gpower_fp1, totalpower_fp1, freq_fp1, power_fp1 = calculate_band(normalize(st.session_state.data_eeg_fp1['EEG Fp1']))
                
                # fig = px.line(x=freq_fp1,
                #             y=power_fp1,
                #             labels={'x': 'Frequency (hHz)', 'y': 'Power'},
                #             title=f'Frequency Domain Fp1')
                # st.plotly_chart(fig)

                mpf_fp1 = mpf(freq_fp1, power_fp1)

                st.session_state.a_power_fp1 = alpha_band_fp1
                st.session_state.b_power_fp1 = beta_band_fp1
                st.session_state.mpf_fp1 = mpf_fp1
                st.session_state.total_power_fp1 = delta_band_fp1 + theta_band_fp1 + alpha_band_fp1 + beta_band_fp1 + gamma_band_fp1

                st.subheader("EEG Fp1 FEATURES")
                feature_data_fp1 = {
                    "Feature": [ "LAP Fp1", "LBP Fp1", "MPF Fp1"],
                     "Value": [alpha_band_fp1/st.session_state.total_power_fp1, beta_band_fp1/st.session_state.total_power_fp1, mpf_fp1]
                }
                st.table(feature_data_fp1)

        elif submenu == "F3":
            st.subheader("EEG F3 Data")
            if "data_eeg_f3" in st.session_state and not st.session_state.data_eeg_f3.empty:
                fig = px.line(x=st.session_state.data_eeg_f3["Elapsed Time"],
                            y=normalize(st.session_state.data_eeg_f3['EEG F3']),
                            labels={'x': 'Elapsed time(s)', 'y': 'Magnitude'}, title='EEG F3 Signal')
                st.plotly_chart(fig)

                selected_level = 8
                a_f3 = [None] * selected_level
                d_f3 = [None] * selected_level

                signal = normalize(st.session_state.data_eeg_f3['EEG F3'])
                for level in range(selected_level):
                    div = 2
                    a_f3[level], d_f3[level] = decomposition(div, signal)
                    signal = normalize(a_f3[level])  

                fig = px.line(x=range(len(d_f3[4])),
                            y=d_f3[4],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Detail Coefficients Level 5, Sinyal Gamma')
                st.plotly_chart(fig)

                fig = px.line(x=range(len(d_f3[5])),
                            y=d_f3[5],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Detail Coefficients Level 6, Sinyal Theta')
                st.plotly_chart(fig)

                fig = px.line(x=range(len(d_f3[6])),
                            y=d_f3[6],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Detail Coefficients Level 7, Sinyal Alpha')
                st.plotly_chart(fig)

                fig = px.line(x=range(len(d_f3[7])),
                            y=d_f3[7],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Detail Coefficients Level 8, Sinyal Beta')
                st.plotly_chart(fig)

                fig = px.line(x=range(len(a_f3[7])),
                            y=a_f3[7],
                            labels={'x': 'Index', 'y': 'Magnitude'},
                            title=f'Approximation Coefficients Level 8, Sinyal Delta')
                st.plotly_chart(fig)


                delta_band_f3 = delta_eeg(a_f3[7])
                theta_band_f3 = theta_eeg(d_f3[7])
                alpha_band_f3 = alpha_eeg(d_f3[6])
                beta_band_f3 = beta_eeg(d_f3[5])
                gamma_band_f3 = gamma_eeg(d_f3[4])

                dpower_f3, tpower_f3, apower_f3, bpower_f3, gpower_f3, totalpower_f3, freq_f3, power_f3 = calculate_band(normalize(st.session_state.data_eeg_f3['EEG F3']))
                
                # fig = px.line(x=freq_f3,
                #             y=power_f3,
                #             labels={'x': 'Frequency (hHz)', 'y': 'Power'},
                #             title=f'Frequency Domain F3')
                # st.plotly_chart(fig)

                mpf_f3 = mpf(freq_f3, power_f3)
                st.session_state.a_power_f3 = alpha_band_f3
                st.session_state.b_power_f3 = beta_band_f3
                st.session_state.mpf_f3 = mpf_f3
                st.session_state.total_power_f3 = delta_band_f3 + theta_band_f3 + alpha_band_f3 + beta_band_f3 + gamma_band_f3
                st.subheader("EEG F3 FEATURES")
                feature_data_f3 = {
                     "Feature": ["LAP F3", "LBP F3", "MPF F3"],
                     "Value": [alpha_band_f3/st.session_state.total_power_f3, beta_band_f3/st.session_state.total_power_f3, mpf_f3]
                }
                st.table(feature_data_f3)

    if choice == "ANN":
        st.header("ANN STRESS DETECTION")

        mean_rr = st.session_state.mean_rr
        mean_hr = st.session_state.mean_hr
        rmssd = st.session_state.rmssd
        pnn50 = st.session_state.pnn50
        sdnn = st.session_state.sdnn

        lf_power = st.session_state.lf_power
        hf_power = st.session_state.hf_power
        total_power = st.session_state.total_power_ecg

        sd1_value = st.session_state.sd1
        sd2_value = st.session_state.sd2

        time_domain_data = {
            "Metric": ["Mean RR Interval (s)", "Mean Heart Rate (bpm)", "RMSSD", "SDNN", "pNN50"],
            "Value": [f"{mean_rr:.4f}", f"{mean_hr:.4f}", f"{rmssd:.4f}", f"{sdnn:.4f}", f"{pnn50:.4f}"]
        }
        df_time_domain = pd.DataFrame(time_domain_data)

        freq_domain_data = {
            "Metric": ["LF/HF Ratio", "Total Power"], #"VLF Power", "LF Power", "HF Power", 
            "Value": [f"{lf_power / hf_power:.8f}", f"{total_power:.6f}"] #f"{vlf_power}", f"{lf_power}", f"{hf_power}",
        }
        df_freq_domain = pd.DataFrame(freq_domain_data)

        non_linear_data = {
            "Metric": ["SD1", "SD2"],
            "Value": [f"{sd1_value:.4f}", f"{sd2_value:.4f}"]
        }
        df_non_linear = pd.DataFrame(non_linear_data)

        st.subheader("Time Domain Analysis")
        st.table(df_time_domain)

        st.subheader("Frequency Domain Analysis")
        st.table(df_freq_domain)

        st.subheader("Non Linear Analysis")
        st.table(df_non_linear)

        a_power_fp1 = st.session_state.a_power_fp1
        b_power_fp1 = st.session_state.b_power_fp1
        mpf_fp1 = st.session_state.mpf_fp1
        total_power_fp1 = st.session_state.total_power_fp1
        
        st.subheader("EEG Fp1 FEATURES")
        feature_data_fp1 = {
            "Feature": ["LAP Fp1", "LBP Fp1", "MPF Fp1"],
            "Value": [a_power_fp1/total_power_fp1, b_power_fp1/total_power_fp1, mpf_fp1]
        }
        st.table(feature_data_fp1)

        a_power_f3 = st.session_state.a_power_f3
        b_power_f3 = st.session_state.b_power_f3
        mpf_f3 = st.session_state.mpf_f3
        total_power_f3 = st.session_state.total_power_f3

        st.subheader("EEG F3 FEATURES")
        feature_data_f3 = {
            "Feature": ["LAP F3", "LBP F3", "MPF F3"],
            "Value": [a_power_f3/total_power_f3, b_power_f3/total_power_f3, mpf_f3]
        }
        st.table(feature_data_f3)

        lf_hf_ratio = lf_power/hf_power
        # lap_fp1 = a_power_fp1/total_power_fp1
        lbp_fp1 = b_power_fp1/total_power_fp1
        # lap_f3 = a_power_f3/total_power_f3
        # lbp_f3 = b_power_f3/total_power_f3

        # Define features in the new order
        features = [
            lf_hf_ratio,    # LF/HF Ratio
            pnn50,          # PNN50
            mean_hr,        # BPM
            mean_rr,        # Mean Interval
            mpf_fp1,        # MPF Fp1
            lbp_fp1,        # LBPFp1
            sd1_value,      # SD1   
    ]

        features_df = pd.DataFrame([features], columns = [
            'LF/HF_Ratio', 'pNN50', 'Mean_HR', 'Mean_RR', 'MPF Fp1', 'LBPFp1', 'SD1'
        ])

        print(features_df)

        # 1. Normalizing features
        scaler = joblib.load(os.path.join('7Features(SMOTE)StandarScaler.bin'))
        features_scaled = scaler.transform(features_df)

        print(features_scaled)

        # 2. Loading pretrained model
        pretrained_model_path = os.path.join("Anova7030SMOTERank7.keras")
        print(pretrained_model_path)
    
        try:
            model = tf.keras.models.load_model(pretrained_model_path)
            predictions = model.predict(features_scaled)
            # 3. Evaluating model on test data

            # 4. Loop through predictions and print predicted class
            st.subheader("Hasil Prediksi:")
            target_names = ["Stress Rendah", "Stress Sedang", "Stress Tinggi"]
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = target_names[predicted_class_index]

            # Display predicted class using streamlit
            if predicted_class_name == "Stress Rendah":
                st.success(predicted_class_name)
            elif predicted_class_name == "Stress Sedang":
                st.warning(predicted_class_name)
            elif predicted_class_name == "Stress Tinggi":
                st.error(predicted_class_name)

        except Exception as e:
            st.error(f"Error loading or using the model: {str(e)}")

    elif choice == "About":
        st.subheader("About")
        st.markdown(
            """
            This web application is built for processing and feature extraction of ECG and EEG signals by YASMIN FAKHIRA ICHSAN BME#6
            """
        )

if __name__ == "__main__":
    main()  