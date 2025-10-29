FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends tzdata build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# se quiser rodar o build do dataset durante a imagem, abaixo executamos o script que está em scripts/
# se o script precisa de internet/credenciais, considere gerar o dataset no host e comentar a linha abaixo
RUN python /app/scripts/build_dataset.py || true

EXPOSE 8050
CMD sh -c "gunicorn -w 4 -b 0.0.0.0:${PORT:-8050} app:server"
