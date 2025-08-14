# Dockerfile
FROM python:3.11

RUN apt-get update && \
    apt-get install -y build-essential libatlas3-base libfreetype6-dev libpng-dev libopenblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
