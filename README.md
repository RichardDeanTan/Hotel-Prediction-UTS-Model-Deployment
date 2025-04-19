# 🏨 Hotel Booking Cancellation Prediction - UTS Model Deployment

Proyek ini memprediksi apakah suatu pemesanan hotel akan dibatalkan atau tidak menggunakan teknik machine learning. Proyek ini dibuat sebagai bagian dari Ujian Tengah Semester (UTS) mata kuliah Model Deployment.

## 📂 Project Structure

- `2702262652_1B.ipynb` — Jupyter Notebook untuk data exploration, preprocessing, model building, dan model fine tuning.
- `2702262652_2B.py` — Python untuk format OOP menggunakan hasil model terbaik dari Jupyther Notebook.
- `2702262652_3B.py` — File Streamlit yang menjalankan antarmuka prediksi.
- `best_model.pkl` — Model Machine Learning terbaik yang sudah di train.
- `transformer_data.pkl` — Pipeline preprocessing untuk transformasi data.
- `requirements.txt` — Daftar dependensi Python yang diperlukan untuk run project.

## 🚀 Cara Run Aplikasi

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
streamlit run app.py
```
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
Richard Dean Tan — Proyek UTS untuk Universitas Bina Nusantara, Mata Kuliah Model Deployment.

## 📝 License
Proyek ini bersifat open-source dan bebas digunakan untuk keperluan edukasi.
