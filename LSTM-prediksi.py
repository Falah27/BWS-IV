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
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# ==============================
# Fungsi-fungsi evaluasi dan helper
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

def create_multivariate_lagged_dataset(features, target, n_lags):
    X, y = [], []
    if len(features) <= n_lags: return np.array(X), np.array(y)
    for i in range(len(features) - n_lags):
        X.append(features[i:(i + n_lags), :])
        y.append(target[i + n_lags])
    return np.array(X), np.array(y)

def build_lstm_model(n_lags, num_features):
    model = Sequential([
        Input(shape=(n_lags, num_features)),
        LSTM(100, return_sequences=True, activation="tanh"), Dropout(0.2),
        LSTM(50, return_sequences=False, activation="tanh"),
        Dense(25, activation="relu"), Dense(1, activation="softplus")
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")
    return model

# --- FUNGSI GRAFIK 1 ---
def plot_tuning_results(df_results, best_lag, output_dir):
    """Membuat dan menyimpan grafik hasil tuning n_lags."""
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["n_lags"], df_results["NSE_validasi"], marker='o', linestyle='-', color='teal')
    plt.axvline(x=best_lag, color='red', linestyle='--', label=f'n_lags Terbaik ({best_lag})')
    plt.title('Performa Model (NSE) pada Set Validasi', fontsize=16)
    plt.xlabel('Jumlah n_lags', fontsize=12)
    plt.ylabel('Nilai NSE', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Grafik_Hasil_Tuning_Lags.png"), dpi=300)
    plt.close()
    print("✅ Grafik hasil tuning telah disimpan.")

# --- FUNGSI GRAFIK 2 ---
def plot_scatter_results(y_true, y_sim, period, n_lags, r_value, output_dir):
    """Membuat dan menyimpan scatter plot perbandingan observasi vs simulasi."""
    r_squared = r_value**2
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_sim, alpha=0.6, edgecolors='k', facecolors='royalblue')
    
    # Menambahkan garis 1:1
    lims = [min(plt.xlim(), plt.ylim()), max(plt.xlim(), plt.ylim())]
    if lims[0] is not None and lims[1] is not None:
        plt.plot(lims, lims, 'r--', linewidth=2, label='Garis 1:1')
    
    plt.title(f'Scatter Plot Hasil {period} (n_lags={n_lags})', fontsize=16)
    plt.xlabel('Observasi TMA (m)', fontsize=12)
    plt.ylabel('Simulasi TMA (m)', fontsize=12)
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Memastikan skala sumbu X dan Y sama
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Grafik_Scatter_Plot_{period}.png"), dpi=300)
    plt.close()
    print(f"✅ Grafik scatter plot untuk periode {period} telah disimpan.")


# --- FUNGSI PREDIKSI BARU ---
def predict_future_steps(model, df_all_data, scaler_rain, scaler_tma, 
                         rain_col, tma_obs_col, n_lags, n_steps=24, 
                         future_rain_assumption=0.0):
    """
    Membuat prediksi autoregresif untuk n_steps ke depan.
    
    PENTING: Fungsi ini mengasumsikan nilai untuk fitur eksogen (curah hujan).
    """
    print(f"\nMemulai prediksi autoregresif untuk {n_steps} langkah ke depan...")
    
    # 1. Siapkan asumsi curah hujan masa depan
    if isinstance(future_rain_assumption, (int, float)):
        future_rain_values = np.full(n_steps, future_rain_assumption)
    elif len(future_rain_assumption) == n_steps:
        future_rain_values = np.array(future_rain_assumption)
    else:
        raise ValueError(f"future_rain_assumption harus berupa angka atau array dengan panjang {n_steps}")
        
    print(f"⚠️  PERINGATAN: Menggunakan asumsi curah hujan = {future_rain_values[0]} mm untuk {n_steps} hari ke depan.")
    
    # Scaling nilai hujan masa depan
    future_rain_scaled = scaler_rain.transform(future_rain_values.reshape(-1, 1)).flatten()
    
    # 2. Ambil data sekuens terakhir yang diketahui dari *seluruh* dataset
    last_sequence_data = df_all_data.iloc[-n_lags:]
    
    # 3. Scaling data sekuens terakhir
    rain_last_seq = last_sequence_data[rain_col].values.reshape(-1, 1)
    tma_last_seq = last_sequence_data[tma_obs_col].values.reshape(-1, 1)
    
    rain_last_seq_scaled = scaler_rain.transform(rain_last_seq)
    tma_last_seq_scaled = scaler_tma.transform(tma_last_seq)
    
    # Gabungkan menjadi fitur: [hujan, tma]
    current_input_sequence = np.hstack([rain_last_seq_scaled, tma_last_seq_scaled]) # Shape (n_lags, 2)
    
    predictions_scaled = []
    
    # 4. Loop prediksi
    for i in range(n_steps):
        # Siapkan input untuk model
        input_for_model = current_input_sequence.reshape(1, n_lags, current_input_sequence.shape[1])
        
        # Prediksi 1 langkah
        predicted_tma_scaled = model.predict(input_for_model, verbose=0) # Shape (1, 1)
        
        # Simpan hasil prediksi (scaled)
        predictions_scaled.append(predicted_tma_scaled[0, 0])
        
        # Siapkan fitur baru untuk langkah berikutnya
        current_future_rain_scaled = future_rain_scaled[i]
        new_features_scaled = np.array([[current_future_rain_scaled, predicted_tma_scaled[0, 0]]]) # Shape (1, 2)
        
        # "Geser" sekuens input: hapus data terlama, tambahkan data terbaru (prediksi)
        current_input_sequence = np.append(current_input_sequence[1:], new_features_scaled, axis=0)
        
    # 5. Inverse transform semua prediksi
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    future_predictions_unscaled = scaler_tma.inverse_transform(predictions_scaled)
    
    print("✅ Prediksi masa depan selesai.")
    return future_predictions_unscaled


# ==========================================================
# FUNGSI UTAMA: Alur Kerja Lengkap (Tuning, Training, Testing, Prediksi)
# ==========================================================
def run_complete_workflow(input_file, sheet_train, sheet_validate, sheet_test, 
                          rain_col, tma_obs_col, output_dir, lags_to_test, 
                          n_future_steps=24): # <-- Parameter baru
    # Tahap 1 & 2 (Tuning Parameter) ... (Sama seperti sebelumnya)
    print("="*60); print("Memulai Alur Kerja Pemodelan Lengkap (Train, Validate, Test)"); print("="*60)
    print("\n--- Tahap 1: Membaca dan Memproses Data ---")
    df_train = pd.read_excel(input_file, sheet_name=sheet_train)
    df_validate = pd.read_excel(input_file, sheet_name=sheet_validate)
    df_test = pd.read_excel(input_file, sheet_name=sheet_test)
    for df in [df_train, df_validate, df_test]: df.interpolate(method='linear', inplace=True, limit_direction='both')
    print(f"Data Train: {len(df_train)} baris\nData Validasi: {len(df_validate)} baris\nData Test: {len(df_test)} baris")
    
    # Gabungkan semua data *sekarang* untuk digunakan nanti di Tahap 5
    df_all_data = pd.concat([df_train, df_validate, df_test], ignore_index=True)
    
    print("\n--- Tahap 2: Mencari n_lags Terbaik Menggunakan Data Validasi ---")
    scaler_rain = MinMaxScaler(); scaler_tma = MinMaxScaler()
    scaler_rain.fit(df_train[rain_col].values.reshape(-1, 1)); scaler_tma.fit(df_train[tma_obs_col].values.reshape(-1, 1))
    rain_train_scaled = scaler_rain.transform(df_train[rain_col].values.reshape(-1, 1)); tma_train_scaled = scaler_tma.transform(df_train[tma_obs_col].values.reshape(-1, 1))
    features_train_scaled = np.hstack([rain_train_scaled, tma_train_scaled])
    rain_val_scaled = scaler_rain.transform(df_validate[rain_col].values.reshape(-1, 1)); tma_val_scaled = scaler_tma.transform(df_validate[tma_obs_col].values.reshape(-1, 1))
    features_val_scaled = np.hstack([rain_val_scaled, tma_val_scaled])
    validation_results = []
    
    for lag in lags_to_test:
        print(f"Menguji n_lags = {lag}...")
        X_train, y_train = create_multivariate_lagged_dataset(features_train_scaled, tma_train_scaled.flatten(), lag)
        X_val, y_val_true_scaled = create_multivariate_lagged_dataset(features_val_scaled, tma_val_scaled.flatten(), lag)
        if len(X_train) == 0 or len(X_val) == 0: print(f"Data tidak cukup untuk n_lags={lag}, melewati."); continue
        model = build_lstm_model(lag, X_train.shape[2])
        model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)])
        sim_val_scaled = model.predict(X_val, verbose=0)
        sim_val = scaler_tma.inverse_transform(sim_val_scaled); y_val_true = scaler_tma.inverse_transform(y_val_true_scaled.reshape(-1, 1))
        nse_val = nse(sim_val.flatten(), y_val_true.flatten())
        validation_results.append({"n_lags": lag, "NSE_validasi": nse_val})
        
    df_val_results = pd.DataFrame(validation_results)
    print("\n--- Hasil Tuning Parameter ---"); print(df_val_results.to_string(index=False))
    best_n_lags = int(df_val_results.loc[df_val_results['NSE_validasi'].idxmax()]['n_lags'])
    print(f"\n✅ Ditemukan n_lags terbaik = {best_n_lags} (berdasarkan performa di set validasi)")
    
    # --- MEMANGGIL FUNGSI GRAFIK TUNING ---
    plot_tuning_results(df_val_results, best_n_lags, output_dir)
    
    # Tahap 3 (Pelatihan Model Final) ... (Sama seperti sebelumnya)
    print("\n--- Tahap 3: Melatih Model Final dengan Parameter Terbaik ---")
    df_combined_train = pd.concat([df_train, df_validate], ignore_index=True)
    scaler_rain_final = MinMaxScaler(); scaler_tma_final = MinMaxScaler()
    scaler_rain_final.fit(df_combined_train[rain_col].values.reshape(-1, 1)); scaler_tma_final.fit(df_combined_train[tma_obs_col].values.reshape(-1, 1))
    rain_comb_scaled = scaler_rain_final.transform(df_combined_train[rain_col].values.reshape(-1, 1)); tma_comb_scaled = scaler_tma_final.transform(df_combined_train[tma_obs_col].values.reshape(-1, 1))
    features_comb_scaled = np.hstack([rain_comb_scaled, tma_comb_scaled])
    rain_test_scaled = scaler_rain_final.transform(df_test[rain_col].values.reshape(-1, 1)); tma_test_scaled = scaler_tma_final.transform(df_test[tma_obs_col].values.reshape(-1, 1))
    features_test_scaled = np.hstack([rain_test_scaled, tma_test_scaled])
    X_train_final, y_train_final = create_multivariate_lagged_dataset(features_comb_scaled, tma_comb_scaled.flatten(), best_n_lags)
    X_test_final, y_test_true_scaled = create_multivariate_lagged_dataset(features_test_scaled, tma_test_scaled.flatten(), best_n_lags)
    print(f"Melatih model final dengan {len(X_train_final)} sampel data gabungan...")
    
    final_model = build_lstm_model(best_n_lags, X_train_final.shape[2])
    final_model.fit(X_train_final, y_train_final, epochs=150, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)])
    print("✅ Model final selesai dilatih.")

    # --- 4. UJIAN AKHIR (FINAL TESTING) ---
    print("\n--- Tahap 4: Ujian Akhir pada Data Test ---")
    sim_test_scaled = final_model.predict(X_test_final, verbose=0)
    sim_test = scaler_tma_final.inverse_transform(sim_test_scaled)
    y_test_true = scaler_tma_final.inverse_transform(y_test_true_scaled.reshape(-1, 1))
    R_final = np.corrcoef(y_test_true.flatten(), sim_test.flatten())[0, 1]
    NSE_final = nse(sim_test.flatten(), y_test_true.flatten())
    RSR_final = rsr(sim_test.flatten(), y_test_true.flatten())
    print("\n" + "*"*60); print("📊 HASIL AKHIR PERFORMA MODEL (PADA DATA TEST) 📊".center(60)); print("*"*60)
    print(f"   Model Final         : LSTM dengan n_lags = {best_n_lags}")
    print(f"   Koefisien Korelasi (R): {R_final:.4f}"); print(f"   NSE                 : {NSE_final:.4f}"); print(f"   RSR                 : {RSR_final:.4f}")
    print("*"*60); print("Ini adalah skor performa model Anda yang paling jujur.")
    df_final_results = pd.DataFrame([{'best_n_lags': best_n_lags, 'R_test': R_final, 'NSE_test': NSE_final, 'RSR_test': RSR_final}])
    df_final_results.to_excel(os.path.join(output_dir, "Hasil_Akhir_Testing.xlsx"), index=False)
    
    # --- MEMANGGIL FUNGSI GRAFIK TIME SERIES & SCATTER PLOT ---
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_true.flatten(), label="Data Observasi (Test)", color="blue")
    plt.plot(sim_test.flatten(), label="Data Simulasi (Test)", color="red", linestyle='--')
    plt.title(f"Hasil Ujian Akhir Model pada Data Test (n_lags = {best_n_lags})", fontsize=16)
    plt.xlabel("Langkah Waktu (Time Step)"); plt.ylabel("TMA (m)")
    plt.legend(); plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(output_dir, "Grafik_Hasil_Akhir_Testing_TimeSeries.png"), dpi=300)
    plt.close()
    print("\n✅ Hasil akhir (Excel dan Grafik Time Series) telah disimpan.")
    
    plot_scatter_results(y_test_true.flatten(), sim_test.flatten(), "Test", best_n_lags, R_final, output_dir)

    # ==========================================================
    # --- TAHAP 5: PREDIKSI MASA DEPAN (BARU) ---
    # ==========================================================
    print("\n--- Tahap 5: Prediksi Masa Depan ---")
    
    # Tentukan asumsi curah hujan. Default: 0 mm untuk semua n_future_steps
    # Anda bisa mengubah ini, misal: [10, 5, 0, 0, ..., 0]
    future_rain_values = np.zeros(n_future_steps) 
    
    future_predictions = predict_future_steps(
        model=final_model,
        df_all_data=df_all_data, # Menggunakan *semua* data untuk sekuens input terakhir
        scaler_rain=scaler_rain_final,
        scaler_tma=scaler_tma_final,
        rain_col=rain_col,
        tma_obs_col=tma_obs_col,
        n_lags=best_n_lags,
        n_steps=n_future_steps,
        future_rain_assumption=future_rain_values
    )
    
    # --- Simpan dan Cetak Hasil Prediksi ---
    df_predictions = pd.DataFrame(future_predictions, columns=["Prediksi_TMA_m"])
    df_predictions.index = pd.RangeIndex(start=1, stop=n_future_steps + 1)
    df_predictions.index.name = "Hari_ke_Depan"
    print("\n--- Hasil Prediksi ---")
    print(df_predictions.to_string())
    df_predictions.to_excel(os.path.join(output_dir, f"Hasil_Prediksi_{n_future_steps}_Hari.xlsx"))
    print(f"\n✅ Hasil prediksi {n_future_steps} hari ke depan telah disimpan ke Excel.")
    
    # --- Buat Grafik Prediksi Masa Depan ---
    
    # Siapkan data historis (data test) untuk digabungkan di grafik
    obs_test_data = y_test_true.flatten()
    sim_test_data = sim_test.flatten()
    
    # Buat sumbu waktu
    history_axis = np.arange(len(obs_test_data))
    future_axis = np.arange(len(obs_test_data), len(obs_test_data) + n_future_steps)
    
    plt.figure(figsize=(17, 8))
    # Plot data historis (test)
    plt.plot(history_axis, obs_test_data, label="Observasi (Test)", color="blue", linewidth=1.5)
    plt.plot(history_axis, sim_test_data, label="Simulasi (Test)", color="red", linestyle='--', linewidth=1.5)
    
    # Plot data prediksi
    plt.plot(future_axis, future_predictions.flatten(), label=f"Prediksi {n_future_steps} Hari", color="green", linestyle='-.', marker='o', markersize=4)
    
    # Garis vertikal pemisah
    plt.axvline(x=history_axis[-1], color='gray', linestyle=':', label='Batas Prediksi')
    
    plt.title(f"Prediksi {n_future_steps} Hari ke Depan (n_lags = {best_n_lags})", fontsize=16)
    plt.xlabel("Langkah Waktu (Hari)"); plt.ylabel("TMA (m)")
    plt.legend(); plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Grafik_Prediksi_Masa_Depan.png"), dpi=300)
    plt.close()
    
    print("✅ Grafik prediksi masa depan telah disimpan.")


# ==============================
# Pemanggilan Utama
# ==============================
if __name__ == "__main__":
    try:
        input_file = r"F:/WORK/SubDAS Sanggai (Split).xlsx"
        output_dir = r"F:/WORK/result_final"
        sheet_train = "Data_Train"; sheet_validate = "Data_Validate"; sheet_test = "Data_Test"
        rain_col = "Curah Hujan Harian Thiessen (mm)"; tma_obs_col = "PDA 6 (m)"
        
        # Menggunakan [20] berarti kita tidak 'mencari' lag, hanya 'menggunakan' lag 20
        lags_to_test = [20]
        
        # --- MODIFIKASI ---
        # Tentukan berapa hari (langkah) ke depan yang ingin diprediksi
        langkah_prediksi = 5
        
        os.makedirs(output_dir, exist_ok=True)
        
        run_complete_workflow(
            input_file, sheet_train, sheet_validate, sheet_test,
            rain_col, tma_obs_col, output_dir, lags_to_test,
            n_future_steps=langkah_prediksi # <-- Melewatkan parameter baru
        )
        
        print("\n" + "="*60); print("🎉 ALUR KERJA SEMPURNA SELESAI 🎉".center(60)); print("="*60)

    except FileNotFoundError:
        print(f"\n❌ ERROR: File tidak ditemukan. Pastikan path atau nama sheet benar.")
    except Exception as e:
        print(f"\n❌ Terjadi error tak terduga: {e}")
