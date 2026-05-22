FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    cmake build-essential git \
    alsa-utils libasound2-dev \
    libsndfile1 libsndfile1-dev \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# whisper.cpp compilado SIN Vulkan (CPU puro — evita bug VRAM allocation issue #2411)
RUN git clone --depth=1 https://github.com/ggml-org/whisper.cpp /whisper.cpp

RUN cd /whisper.cpp && \
    cmake -B build \
      -DGGML_VULKAN=OFF \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j8 --target whisper-cli

COPY docker/brain/requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

# Piper TTS binario precompilado para x86_64 Linux
RUN curl -fsSL "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz" \
    | tar xz -C /usr/local/ && \
    ln -s /usr/local/piper/piper /usr/local/bin/piper

COPY scripts/ /app/scripts/

WORKDIR /app

ENV WHISPER_BIN=/whisper.cpp/build/bin/whisper-cli
ENV WHISPER_MODEL=/models/ggml-small.bin
ENV LLAMA_HOST=127.0.0.1
ENV LLAMA_PORT=8080
ENV PIPER_BIN=/usr/local/bin/piper
ENV PIPER_MODEL=/voices/es_ES-davefx-medium.onnx
ENV OWW_MODEL=/models/coramo.onnx
ENV MANAGED_LLM=false
ENV CORAMO_LOG=/tmp/coramo-debug.log

CMD ["python3.11", "scripts/coramo-assistant.py"]
