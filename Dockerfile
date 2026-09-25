# GEO-MN — supply-continuity decision support API + static frontend
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user; decision log goes to a writable directory.
RUN useradd --create-home geomn && mkdir -p /app/data/runtime && chown -R geomn /app/data/runtime
USER geomn

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
