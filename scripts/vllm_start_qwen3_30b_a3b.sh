#!/bin/bash
/home/kdsoft/miniconda3/envs/vllm-0.9.0/bin/vllm serve "/data/models/Qwen3-30B-A3B" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.45 \
  --port 8000 \
  --max-model-len 32768 \
  --served-model-name "Qwen3-30B-A3B" \
  --api-key ${LLM_API_KEY:-your_api_key_here} \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
