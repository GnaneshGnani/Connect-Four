FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    gymnasium \
    pygame \
    numpy \
    pandas \
    matplotlib \
    tqdm

COPY *.py .

RUN mkdir -p /app/models

COPY models/agent.pth /app/models/agent.pth
COPY models/agent_config.json /app/models/agent_config.json

CMD ["python", "main.py", "--no-train"]