FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake build-essential git \
    libvulkan-dev vulkan-tools \
    mesa-vulkan-drivers \
    spirv-headers spirv-tools glslang-tools glslc \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp /llama.cpp

# Limitar a 8 jobs para no agotar RAM durante el build
RUN cd /llama.cpp && \
    cmake -B build \
      -DGGML_VULKAN=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j8 --target llama-server

EXPOSE 8080

ENTRYPOINT ["/llama.cpp/build/bin/llama-server"]
CMD ["--model", "/models/Qwen3-8B-Q5_K_M.gguf", \
     "--n-gpu-layers", "999", \
     "--ctx-size", "2048", \
     "--cache-type-k", "q8_0", \
     "--cache-type-v", "q8_0", \
     "--parallel", "1", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--log-disable"]
