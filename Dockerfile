# Build stage - install all dependencies
FROM python:3.11 AS builder

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install deps in venv for easier copying
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# Runtime stage - smaller final image
FROM python:3.11-slim

WORKDIR /app

# Grab the venv from builder
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Copy app files
COPY scripts/ ./scripts/
COPY serving/ ./serving/
COPY dependencies/ ./dependencies/
COPY data/ ./data/

EXPOSE 5001

CMD ["python", "serving/serve.py"]