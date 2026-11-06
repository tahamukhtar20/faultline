FROM python:3.13-slim

RUN pip install --no-cache-dir --no-compile -r /dev/stdin <<'EOF'
fastapi==0.162.0
httpx==0.28.1
pytest==9.1.1
uvicorn==0.54.0
EOF

RUN useradd --uid 10001 --no-create-home --no-log-init sandbox \
    && mkdir /data /workspace \
    && chown sandbox /data /workspace

COPY toolbox.py provider.py crash_api.py /opt/faultline/

USER sandbox
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace
