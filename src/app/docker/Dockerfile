FROM python:3.11-slim

WORKDIR /app

RUN /usr/bin/apt-get update && \
    /usr/bin/apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY src/app/docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/app/ /app/

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=30s --start-period=10s CMD curl -f http://localhost:80/health || exit 1

CMD ["uvicorn", "freq:app", "--host", "0.0.0.0", "--port", "80"]