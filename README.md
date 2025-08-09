# Monolith
An AI cluster in the ONI format


# ONI Monolith (Local Omnimodel)

This repo wires together permissively-licensed Hugging Face models into a single agent runner.

## Quickstart
1. Create a venv and install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Run chat
   ```python
   main.py --task chat --prompt "Plan→act→check in one paragraph"
   ```
3. Start API server
   ```
   uvicorn run.oni_serve:app ==reload
```

Env overrides

Set environment variables to swap mo3dels:

ONI_TEXT_MODEL (default Qwen/Qwen2.5-7B-Instruct)

ONI_VISION_MODEL (Qwen/Qwen2-VL-7B-Instruct)

ONI_CODER_MODEL (DeepSeek-Coder-V2-Instruct)

ONI_FUNC_MODEL (Functionary v3.x)

ONI_ASR_MODEL  (Whisper large-v3)

ONI_EMBED_MODEL_1 (BAAI/bge-m3)

ONI_EMBED_MODEL_2 (nomic-ai/nomic-embed-text-v1.5)
