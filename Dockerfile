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

# Configure OpenCode permissions (deny tool execution — see README)
RUN mkdir -p /root/.config/opencode && \
    echo '{"permissions":{"*":"deny"}}' > /root/.config/opencode/opencode.json

# Install Python + proxy deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip3 install --break-system-packages -r /app/requirements.txt

COPY proxy.py /app/

WORKDIR /sandbox
EXPOSE 8000

ENTRYPOINT ["python3", "/app/proxy.py", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--mode", "cli"]
