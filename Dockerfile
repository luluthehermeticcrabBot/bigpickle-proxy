FROM debian:bookworm-slim

# Install Bun (OpenCode runtime dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip ca-certificates \
    && curl -fsSL https://bun.sh/install | bash \
    && mv /root/.bun/bin/bun /usr/local/bin/bun \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCode
RUN curl -fsSL https://opencode.ai/install | bash \
    && mv /root/.opencode/bin/opencode /usr/local/bin/opencode

# Configure OpenCode (permission deny happens at session level via API)
RUN mkdir -p /root/.config/opencode

# Install Python + proxy deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip3 install --break-system-packages -r /app/requirements.txt

COPY proxy.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

WORKDIR /sandbox
EXPOSE 8000 4096

ENTRYPOINT ["/app/entrypoint.sh"]
