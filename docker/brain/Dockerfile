FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip \
    cmake build-essential git \
    alsa-utils libasound2-dev \
    libsndfile1 libsndfile1-dev \
    espeak-ng \
    libvulkan-dev mesa-vulkan-drivers \
    spirv-headers spirv-tools glslang-tools glslc \
    && rm -rf /var/lib/apt/lists/*

# whisper.cpp compilado CON Vulkan — renderD128 dedicado a Whisper
RUN git clone --depth=1 https://github.com/ggml-org/whisper.cpp /whisper.cpp

RUN cd /whisper.cpp && \
    cmake -B build \
      -DGGML_VULKAN=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j8 --target whisper-cli --target whisper-server

COPY docker/brain/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Piper TTS binario precompilado (descargado al host antes del build)
COPY docker/brain/piper/piper /usr/local/piper/
ENV LD_LIBRARY_PATH=/usr/local/piper:$LD_LIBRARY_PATH

COPY scripts/ /app/scripts/
COPY docker/brain/start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app

ENV WHISPER_BIN=/whisper.cpp/build/bin/whisper-cli
ENV WHISPER_SERVER_BIN=/whisper.cpp/build/bin/whisper-server
ENV WHISPER_SERVER_URL=http://127.0.0.1:8081
ENV WHISPER_MODEL=/models/ggml-small.bin
ENV LLAMA_HOST=127.0.0.1
ENV LLAMA_PORT=8080
ENV PIPER_BIN=/usr/local/piper/piper
ENV PIPER_MODEL=/voices/es_ES-davefx-medium.onnx
ENV OWW_MODEL=/models/coramo.onnx
ENV MANAGED_LLM=false
ENV CORAMO_LOG=/tmp/coramo-debug.log

CMD ["/app/start.sh"]
