# Python tabanlı resmi Docker imajını kullanıyoruz
FROM python:3.10-slim

# Sistem bağımlılıklarını yükleyin
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    libsndfile1 \
    && apt-get clean

# Python bağımlılıklarını yüklemek için pip kullanıyoruz
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Çalışma dizinini ayarlayın
WORKDIR /app

# Model dosyasını kopyalayın
COPY ANN.pkl /app/ANN.pkl

# Veritabanı dosyasını kopyalayın
COPY analysis_results.db /app/analysis_results.db

# Proje dosyalarını kopyalayın
COPY . /app

# Uygulamayı başlatmak için komut
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
