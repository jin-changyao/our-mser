# Server run commands

## Extra dependencies

先在已有 `our-mser` 环境中检查：

```bash
python - <<'PY'
import importlib.util
for name in ["torch", "transformers", "accelerate", "datasets", "peft", "trl", "lightning", "bitsandbytes", "sentence_transformers", "faiss", "flash_attn"]:
    print(f"{name}: {bool(importlib.util.find_spec(name))}")
PY
```

这个复现代码相对于现有环境主要新增：

```bash
pip install trl lightning bitsandbytes sentence-transformers faiss-cpu
```

`flash-attn` 需要与服务器的 PyTorch、CUDA 和编译器匹配，不能盲目安装。它已单独放在 `requirements_server_optional_flashattn.txt`。代码默认请求 `flash_attention_2`；如果服务器安装失败，可以把训练代码切换到 PyTorch `sdpa`，功能不变但速度可能下降。必需依赖清单见 `requirements_server_extra.txt`。

使用 SDPA fallback：

```bash
PRC_EMO_ATTN_IMPL=sdpa bash scripts/run_reproduction_3gpu.sh
```

在服务器上执行：

```bash
cd /home/pc/jcy/Our-MSER/PRC-Emo-reproduction
bash scripts/run_reproduction_3gpu.sh
```

默认行为：

1. GPU 0 顺序生成 IEMOCAP 和 MELD 的 Qwen2.5 情绪描述。
2. GPU 0 顺序生成 IEMOCAP 和 MELD 的 Qwen2.5 说话人特征。
3. GPU 0 根据 Hugging Face 检索库生成两套最终 Prompt JSONL。
4. GPU 0、1、2 并行训练 IEMOCAP，使用 seed 42、43、44。
5. IEMOCAP 三个任务全部结束后，GPU 0、1、2 并行训练 MELD。
6. 写出 `runs/RESULTS.md`、`runs/run_status.tsv` 和各任务日志。

如果三张卡的编号是 0、1、3：

```bash
GPU0=0 GPU1=1 GPU2=3 bash scripts/run_reproduction_3gpu.sh
```

如果服务器上的工程路径不同：

```bash
PROJECT_ROOT=/your/path/PRC-Emo-reproduction \
MODEL_PATH=/home/pc/jcy/models/Qwen2.5-7B-Instruct \
bash scripts/run_reproduction_3gpu.sh
```

服务器无法访问 Hugging Face 时，先把本地的 `models/all-MiniLM-L6-v2/` 上传到复现目录：

```bash
scp -r models/all-MiniLM-L6-v2 \
  user@server:/home/pc/jcy/Our-MSER/PRC-Emo-reproduction/models/
```

也可以使用：

```bash
rsync -av models/all-MiniLM-L6-v2/ \
  user@server:/home/pc/jcy/Our-MSER/PRC-Emo-reproduction/models/all-MiniLM-L6-v2/
```

代码默认从 `./models/all-MiniLM-L6-v2` 加载，也可以显式指定：

```bash
PRC_EMO_EMBEDDING_MODEL_PATH=/home/pc/jcy/Our-MSER/PRC-Emo-reproduction/models/all-MiniLM-L6-v2 \
bash scripts/run_reproduction_3gpu.sh
```

作者检索库约 402 MB，也超过 GitHub 单文件限制，需要单独上传：

```bash
scp data/Emotion_Retrieval_Library_3.json \
  user@server:/home/pc/jcy/Our-MSER/PRC-Emo-reproduction/data/
```

GitHub 中保留检索库的 README 和相关代码，但不提交这个大文件本身。

单独运行生成阶段：

```bash
cd /home/pc/jcy/Our-MSER/PRC-Emo-reproduction

CUDA_VISIBLE_DEVICES=0 PRC_EMO_DATASET=iemocap \
PRC_EMO_DATA_FOLDER=$PWD/data \
PRC_EMO_MODEL_PATH=/home/pc/jcy/models/Qwen2.5-7B-Instruct \
python src/llm_emotion_extract_v2.py

CUDA_VISIBLE_DEVICES=0 PRC_EMO_DATASET=iemocap \
PRC_EMO_DATA_FOLDER=$PWD/data \
PRC_EMO_MODEL_PATH=/home/pc/jcy/models/Qwen2.5-7B-Instruct \
python src/llm_bio_extract_v2.py

CUDA_VISIBLE_DEVICES=0 python scripts/generate_prompt_data.py --dataset iemocap
```

把 `iemocap` 改成 `meld` 即可处理 MELD。

失败样本会写入每个训练任务目录中的 `result_*.json` 的 `failure_samples` 字段，同时汇总到 `runs/RESULTS.md`。每条失败样本包含当前句子、完整 Prompt、messages、金标准、预测标签和模型原始回答。
