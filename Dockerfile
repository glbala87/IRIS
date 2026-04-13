FROM python:3.11-slim

LABEL maintainer="IRIS developers"
LABEL description="IRIS: Isoform-Resolved Inference for Single-cells"
LABEL version="0.1.0"

# System deps for HDF5, matplotlib, and bioinformatics tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    pkg-config \
    gcc \
    g++ \
    samtools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install IRIS with all extras
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[all]"

# Verify installation
RUN iris --version

ENTRYPOINT ["iris"]
CMD ["--help"]
