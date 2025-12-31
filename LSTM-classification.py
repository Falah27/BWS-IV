import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import tf2onnx # <--- IMPORT BARU

# ==============================
# 1. Fungsi evaluasi (Tidak ada perubahan)
# ==============================
def nse(sim, obs):
    if np.std(obs) == 0: return -np.inf
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def rmse(sim, obs):
    return np.sqrt(np.mean((obs - sim) ** 2))

def rsr(sim, obs):
    std_obs = np.std(obs)
    if std_obs == 0: return np.inf
    return rmse(sim, obs) / std_obs

# ==============================
# 2. Fungsi pembuat dataset (Tidak ada perubahan)
# ==============================
def create_multivariate_lagged_dataset(features, target, n_lags):
    X, y = [], []
    for i in range(len(features) - n_lags):
        X.append(features[i:(i + n_lags), :])
        y.append(target[i + n_lags])
    return np.array(X), np.array(y)

# ==============================
# 3. Fungsi smoothing (Tidak ada perubahan)
# ==============================
def smooth_curve(data, window=3):
    return np.convolve(data, np.ones(window)/window, mode="same")

# ==============================
# 4. Fungsi Pembangun Model LSTM (Tidak ada perubahan)
# ==============================
def build_lstm_model(n_lags, num_features):
    model = Sequential([
        Input(shape=(n_lags, num_features)),
        LSTM(100, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        LSTM(50, return_sequences=False, activation="tanh"),
        Dense(25, activation="relu"),
        Dense(1, activation="softplus")
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")
    return model

# ==========================================================
# 5. FUNGSI BARU: Klasifikasi Status Berdasarkan TMA
# ==========================================================
def classify_tma(tma_values, thresholds):
    """
    Mengklasifikasikan nilai TMA ke dalam kategori status bahaya.
    Ambang batas (thresholds) didasarkan pada kategori curah hujan dari BMKG
    yang disesuaikan dengan level TMA.

    Kategori berdasarkan curah hujan:
      - Berawan - Hujan Ringan: Aman
      - Sedang: Waspada
      - Lebat: Siaga
      - Sangat Lebat - Ekstrem: Bahaya

    Args:
        tma_values (np.array): Array 1D berisi nilai-nilai TMA hasil prediksi.
        thresholds (dict): Dictionary berisi ambang batas TMA untuk setiap status.
                           Contoh: {'waspada': 2.0, 'siaga': 3.5, 'bahaya': 5.0}

    Returns:
        list: Sebuah list berisi label status untuk setiap nilai TMA.
    """
    labels = []
    # Urutan pengecekan harus dari yang tertinggi ke terendah
    for tma in tma_values:
        if tma >= thresholds['bahaya']:
            labels.append("Bahaya")
        elif tma >= thresholds['siaga']:
            labels.append("Siaga")
        elif tma >= thresholds['waspada']:
            labels.append("Waspada")
        else:
            labels.append("Aman")
    return labels

# ==========================================================
# 6. FUNGSI UTAMA: Proses Training dan Testing (Sudah Dimodifikasi)
# ==========================================================
def run_validation_process(input_file, sheet_cal, sheet_val, rain_col, tma_obs_col, tma_thresholds, output_dir, n_lags):
    print("=" * 60)
    print("Memulai Proses Kalibrasi dan Validasi Model")
    print(f"Menggunakan n_lags = {n_lags}")
    print("=" * 60)

    # ==============================
    # TAHAP 1: KALIBRASI (TRAINING MODEL)
    # ==============================
    print("\n--- TAHAP 1: Membaca dan Melatih Model dengan Data Kalibrasi ---")
    df_cal = pd.read_excel(input_file, sheet_name=sheet_cal)
    
    # Pra-pemrosesan data kalibrasi
    df_cal.interpolate(method='linear', inplace=True)
    df_cal.fillna(method='bfill', inplace=True)
    df_cal.fillna(method='ffill', inplace=True)

    # Menyiapkan scaler HANYA berdasarkan data kalibrasi
    scaler_rain = MinMaxScaler()
    scaler_tma = MinMaxScaler()

    # Scaling data kalibrasi
    rain_cal_scaled = scaler_rain.fit_transform(df_cal[rain_col].values.reshape(-1, 1))
    tma_cal_scaled = scaler_tma.fit_transform(df_cal[tma_obs_col].values.reshape(-1, 1))
    
    features_cal_scaled = np.hstack([rain_cal_scaled, tma_cal_scaled])
    num_features = features_cal_scaled.shape[1]
    
    # Membuat dataset untuk training
    X_cal, y_cal = create_multivariate_lagged_dataset(features_cal_scaled, tma_cal_scaled.flatten(), n_lags)

    # Membangun dan melatih model
    model = build_lstm_model(n_lags, num_features)
    early_stopping = EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6)
    
    print(f"Training model dengan {len(X_cal)} sampel data kalibrasi...")
    model.fit(X_cal, y_cal, epochs=150, batch_size=32, verbose=0, callbacks=[early_stopping, reduce_lr])
    print("✅ Model berhasil dilatih.")

    # ==============================================================
    # === BAGIAN BARU: MENYIMPAN MODEL KE FORMAT ONNX ===
    # ==============================================================
    print("\n--- Mengekspor Model ke Format ONNX ---")
    onnx_filepath = os.path.join(output_dir, f"Model_LSTM_Lags_{n_lags}.onnx")
    
    # Menentukan signature input untuk konversi
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input_1")]
    
    # Melakukan konversi
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    
    # Menyimpan file .onnx
    with open(onnx_filepath, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ Model ONNX berhasil disimpan di: {onnx_filepath}")
    # ==============================================================
    # === AKHIR BAGIAN BARU ===
    # ==============================================================

    # ==============================
    # TAHAP 2: VALIDASI (TESTING MODEL)
    # ==============================
    print("\n--- TAHAP 2: Menguji Model dengan Data Validasi ---")
    df_val = pd.read_excel(input_file, sheet_name=sheet_val)

    # Pra-pemrosesan data validasi
    df_val.interpolate(method='linear', inplace=True)
    df_val.fillna(method='bfill', inplace=True)
    df_val.fillna(method='ffill', inplace=True)

    # Scaling data validasi MENGGUNAKAN SCALER DARI DATA KALIBRASI
    rain_val_scaled = scaler_rain.transform(df_val[rain_col].values.reshape(-1, 1))
    tma_val_scaled = scaler_tma.transform(df_val[tma_obs_col].values.reshape(-1, 1))
    
    features_val_scaled = np.hstack([rain_val_scaled, tma_val_scaled])
    
    # Membuat dataset untuk validasi
    X_val, y_val = create_multivariate_lagged_dataset(features_val_scaled, tma_val_scaled.flatten(), n_lags)

    # Melakukan prediksi pada data validasi
    print(f"Melakukan prediksi pada {len(X_val)} sampel data validasi...")
    sim_val_scaled = model.predict(X_val)

    # Mengembalikan nilai ke skala asli
    sim_val = scaler_tma.inverse_transform(sim_val_scaled)
    y_val_true = scaler_tma.inverse_transform(y_val.reshape(-1, 1))
    sim_val_smooth = smooth_curve(sim_val.flatten()).reshape(-1, 1)
    print("✅ Prediksi selesai.")

    # ==============================================================
    # TAHAP 3: EVALUASI, KLASIFIKASI & PENYIMPANAN HASIL VALIDASI
    # ==============================================================
    print("\n--- TAHAP 3: Evaluasi, Klasifikasi, dan Penyimpanan Hasil Validasi ---")
    
    # Hitung metrik performa
    R_val = np.corrcoef(y_val_true.flatten(), sim_val_smooth.flatten())[0, 1]
    NSE_val = nse(sim_val_smooth.flatten(), y_val_true.flatten())
    RSR_val = rsr(sim_val_smooth.flatten(), y_val_true.flatten())

    print("\n📊 HASIL EVALUASI VALIDASI 📊")
    print(f"   R   = {R_val:.4f}")
    print(f"   NSE = {NSE_val:.4f}")
    print(f"   RSR = {RSR_val:.4f}")

    # Membuat dan menyimpan grafik (tidak ada perubahan)
    plt.figure(figsize=(15, 7))
    plt.plot(y_val_true.flatten(), label="Data Observasi (Validasi)", color="blue", linewidth=1.5)
    plt.plot(sim_val_smooth.flatten(), label="Data Simulasi (Validasi)", color="red", linestyle='--', linewidth=1.5)
    plt.title(f"Hasil Validasi Model (n_lags = {n_lags})", fontsize=16)
    plt.xlabel("Langkah Waktu (Time Step)", fontsize=12)
    plt.ylabel("TMA (m)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    metrics_text = f"R   = {R_val:.4f}\nNSE = {NSE_val:.4f}\nRSR = {RSR_val:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    graph_filepath = os.path.join(output_dir, f"Grafik_Hasil_Validasi_Lags_{n_lags}.png")
    plt.savefig(graph_filepath, dpi=300)
    plt.close()
    print(f"\n✅ Grafik validasi berhasil disimpan di: {graph_filepath}")

    # Menyimpan hasil metrik ke Excel
    df_hasil_validasi = pd.DataFrame([{"n_lags": n_lags, "R_validasi": R_val, "NSE_validasi": NSE_val, "RSR_validasi": RSR_val}])
    summary_filepath = os.path.join(output_dir, f"Hasil_Metrik_Validasi_Lags_{n_lags}.xlsx")
    df_hasil_validasi.to_excel(summary_filepath, index=False)
    print(f"✅ Hasil metrik validasi berhasil disimpan di: {summary_filepath}")

    # === BAGIAN BARU: KLASIFIKASI DAN PENYIMPANAN HASIL DETAIL ===
    print("\n--- Melakukan Klasifikasi Status Berdasarkan Prediksi TMA ---")
    status_prediksi = classify_tma(sim_val_smooth.flatten(), tma_thresholds)

    # Siapkan DataFrame untuk menyimpan hasil detail
    # Pastikan data validasi Anda memiliki kolom 'Tanggal' atau kolom waktu lainnya
    # Jika tidak ada, baris ini bisa di-comment dan index numerik akan digunakan
    time_index = df_val.index[n_lags:] # Mengambil index yang sesuai dengan data prediksi

    df_detail_output = pd.DataFrame({
        'Time_Step': time_index,
        'TMA_Observasi (m)': y_val_true.flatten(),
        'TMA_Prediksi (m)': sim_val_smooth.flatten(),
        'Status_Prediksi': status_prediksi
    })

    # Menyimpan DataFrame detail ke file Excel baru
    detail_filepath = os.path.join(output_dir, f"Hasil_Detail_Klasifikasi_Lags_{n_lags}.xlsx")
    df_detail_output.to_excel(detail_filepath, index=False)
    print(f"✅ Hasil detail prediksi dan klasifikasi berhasil disimpan di: {detail_filepath}")
    
# ==============================
# 7. Pemanggilan Utama
# ==============================
if __name__ == "__main__":
    try:
        # --- PENTING: ATUR PARAMETER DI BAWAH INI ---
        
        # 1. Path file dan folder
        input_file = r"F:/WORK/SubDAS Sanggai.xlsx"
        output_dir = r"F:/WORK/result2/class/PDA 6 Thiessen - Sanggai"

        # 2. Nama sheet untuk kalibrasi dan validasi
        sheet_cal = "Data Curah Hujan dan TMA" # Sheet untuk melatih model
        sheet_val = "Data Validasi"            # GANTI DENGAN NAMA SHEET VALIDASI ANDA

        # 3. Nama kolom
        rain_col = "Curah Hujan Harian Thiessen (mm)"
        tma_obs_col = "PDA 6 (m)"

        # 4. n_lags optimal yang sudah ditentukan
        OPTIMAL_N_LAGS = 20

        # 5. (BARU) AMBANG BATAS (THRESHOLDS) KLASIFIKASI TMA
        TMA_THRESHOLDS = {
            'waspada': 2.0,  # Jika TMA >= 2.0 m -> Waspada
            'siaga':   3.5,  # Jika TMA >= 3.5 m -> Siaga
            'bahaya':  5.0   # Jika TMA >= 5.0 m -> Bahaya
            # Nilai di bawah 'waspada' otomatis akan menjadi 'Aman'
        }

        # Pastikan folder output ada
        os.makedirs(output_dir, exist_ok=True)
        
        # Jalankan proses
        run_validation_process(
            input_file=input_file,
            sheet_cal=sheet_cal,
            sheet_val=sheet_val,
            rain_col=rain_col,
            tma_obs_col=tma_obs_col,
            tma_thresholds=TMA_THRESHOLDS, # Parameter baru ditambahkan
            output_dir=output_dir,
            n_lags=OPTIMAL_N_LAGS
        )

        print("\n" + "="*60)
        print("✅ PROSES VALIDASI & KLASIFIKASI SELESAI ✅".center(60))
        print("="*60)

    except FileNotFoundError:
        print(f"\n❌ ERROR: File tidak ditemukan. Pastikan path file benar:\n{input_file}")
    except KeyError as e:
        print(f"\n❌ ERROR: Kolom atau Sheet tidak ditemukan. Detail: {e}")
    except Exception as e:
        print(f"\n❌ Terjadi error tak terduga: {e}")