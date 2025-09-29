# SAR Drone AI Project 

## English 

This project is an **AI-powered Search and Rescue (SAR) Drone** equipped with **Face and Hand Detection** capabilities.  
It helps locate missing persons by detecting **faces and hand signals** in real-time using computer vision, machine learning, and drone navigation.

🔗 **GitHub Repository:** [SAR Drone AI Project](https://github.com/theofrolicdean/sar_drone_ai_project)

---

## Features
- Real-time **Face Detection**   
- **Hand Signal Recognition** 
- **Drone Navigation & Control System**   
- **Simulation Mode** for testing without a drone  
- **Alarm Sound Effects** for alerts 
- Web-based GUI for monitoring  

---

## Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/theofrolicdean/sar_drone_ai_project.git
cd sar_drone_ai_project
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate    # For Windows
```

### 3️⃣ Install Dependencies
You can install dependencies with pip:
```bash
pip install -r requirements.txt
```
Or use Conda (recommended for GPU support):
```bash
conda create --name sar_drone python=3.9
conda activate sar_drone
pip install -r conda_env_requirements.txt
```

### 4️⃣ Run the Project
```bash
python main.py
```
To launch the web GUI:
```bash
python web_gui.py
```

---

## 📂 Project Structure
```
sar_drone_ai_project/
│── control/              # Drone control & joystick integration
│── models/               # Pre-trained AI models
│── navigation/           # Navigation algorithms
│── outputs/              # Output logs & saved results
│── sim/                  # Simulation environment for testing
│── static/               # Static assets (CSS, JS, media)
│── templates/            # Web GUI templates
│── utils/                # Helper functions
│── vision/               # Face & hand detection modules
│── .gitignore            # Git ignore rules
│── config.py             # Configuration file
│── conda_env_requirements.txt  # Conda environment dependencies
│── requirements.txt      # Python dependencies
│── main.py               # Main entry point
│── web_gui.py            # Web GUI entry point
│── train.ipynb           # Training notebook
│── README.md             # Project guide
```

---

## Usage
- Run `python main.py` to start face & hand detection web app
- In **simulation mode**, test the system without a physical drone  
- Alarm sound will activate when detection occurs 

---

## Bahasa Indonesia

Proyek ini adalah **Drone Pencarian dan Penyelamatan (SAR) berbasis AI** dengan kemampuan **Deteksi Wajah dan Tangan**.  
Drone ini membantu menemukan orang hilang dengan mendeteksi **wajah dan sinyal tangan** secara real-time menggunakan computer vision, machine learning, dan sistem navigasi drone.

🔗 **Repositori GitHub:** [SAR Drone AI Project](https://github.com/theofrolicdean/sar_drone_ai_project)

---

## Fitur
- Deteksi **Wajah** secara real-time   
- **Pengenalan Sinyal Tangan** 
- **Sistem Navigasi & Kontrol Drone** 
- **Mode Simulasi** untuk pengujian tanpa drone  
- **Efek Suara Alarm** sebagai peringatan 
- GUI berbasis web untuk pemantauan  

---

## 🛠️ Instalasi & Setup

### 1️⃣ Clone Repositori
```bash
git clone https://github.com/theofrolicdean/sar_drone_ai_project.git
cd sar_drone_ai_project
```

### 2️⃣ Buat Virtual Environment (Disarankan)
```bash
python -m venv venv
source venv/bin/activate   # Untuk Linux/Mac
venv\Scripts\activate    # Untuk Windows
```

### 3️⃣ Install Dependencies
Dengan pip:
```bash
pip install -r requirements.txt
```
Atau dengan Conda (disarankan untuk GPU):
```bash
conda create --name sar_drone python=3.9
conda activate sar_drone
pip install -r conda_env_requirements.txt
```

### 4️⃣ Jalankan Proyek
```bash
python main.py
```
Untuk menjalankan web GUI:
```bash
python web_gui.py
```

---

## 📂 Struktur Proyek
```
sar_drone_ai_project/
│── control/              # Kontrol drone & joystick
│── models/               # Model AI pre-trained
│── navigation/           # Algoritma navigasi
│── outputs/              # Log output & hasil
│── sim/                  # Lingkungan simulasi
│── static/               # Aset statis (CSS, JS, media)
│── templates/            # Template untuk Web GUI
│── utils/                # Fungsi pendukung
│── vision/               # Modul deteksi wajah & tangan
│── .gitignore            # File gitignore
│── config.py             # File konfigurasi
│── conda_env_requirements.txt  # Dependensi Conda
│── requirements.txt      # Dependensi Python
│── main.py               # Entry point utama
│── web_gui.py            # Entry point Web GUI
│── train.ipynb           # Notebook training
│── README.md             # Panduan proyek
```

---

## Cara Penggunaan
- Jalankan `python main.py` untuk memulai deteksi wajah & tangan dalam bentuk aplikasi browser 
- Pada **mode simulasi**, sistem dapat diuji tanpa drone fisik  
- Alarm akan berbunyi saat deteksi terjadi 🚨  

---
