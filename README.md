# 🏨 Hotel Booking Cancellation Prediction - UTS Model Deployment

Proyek ini memprediksi apakah suatu pemesanan hotel akan dibatalkan atau tidak menggunakan teknik machine learning. Proyek ini dibuat sebagai bagian dari Ujian Tengah Semester (UTS) mata kuliah Model Deployment.

## 📂 Project Structure

- `2702262652_1B.ipynb` — Jupyter Notebook untuk data exploration, preprocessing, model building, dan model fine tuning.
- `2702262652_2B.py` — Python untuk format OOP menggunakan hasil model terbaik dari Jupyther Notebook.
- `2702262652_3B.py` — Python untuk inference model yang sudah dibuat.
- `2702262652_4B.py` — File Streamlit yang menjalankan website prediksi.
- `best_model.pkl` — Model Machine Learning terbaik yang sudah di train.
- `transformer_data.pkl` — Pipeline preprocessing untuk transformasi data.
- `requirements.txt` — Daftar dependensi Python yang diperlukan untuk run project.

## 🚀 Cara Run Aplikasi

### 🔹 1. Jalankan Secara Lokal
### Clone Repository
```bash
git clone https://github.com/RichardDeanTan/Hotel-Prediction-UTS-Model-Deployment.git
cd Hotel-Prediction-UTS-Model-Deployment
```
### Install Dependensi
```bash
pip install -r requirements.txt
```
### Jalankan Aplikasi Streamlit
```bash
streamlit run 2702262652_4B.py
```

### 🔹 2. Jalankan Secara Online (Tidak Perlu Install)
Klik link berikut untuk langsung membuka aplikasi web:
#### 👉 [Aplikasi Prediksi Hotel](https://hotel-prediction-uts-md-2702262652.streamlit.app/)

## 💡 Fitur
- Memprediksi apakah suatu pemesanan hotel akan dibatalkan.
- Interactive web interface menggunakan Streamlit.
- Model and transformer yang digunakan untuk prediksi secara real time.

## ⚙️ Tech Stack
- Python
- Streamlit
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- Joblib

## 🧠 Model
Model dikembangkan di dalam Jupyther Notebook (Hotel Bookings UTS.ipynb) dengan
- Data preprocessing pipeline (transformer_data.pkl)
- Final model (best_model.pkl)

## 👨‍💻 Pembuat
Richard Dean Tanjaya — Proyek UTS Mata Kuliah Model Deployment, Universitas Bina Nusantara

## 📝 License
Proyek ini bersifat open-source dan bebas digunakan untuk keperluan edukasi.
