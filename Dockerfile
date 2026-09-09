FROM python:3.12-slim

WORKDIR /app

# Install deps first so Docker can cache this layer between builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY cpu.py cpu_server.py cpu_visualiser_simple.html ./

# Fly sets PORT at runtime; default to 8080 for local docker runs
ENV PORT=8080
EXPOSE 8080

# gunicorn, not the Flask dev server, for production.
# cpu_server.py defines the Flask instance as `app`, so the target is cpu_server:app
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 60 cpu_server:app"]
