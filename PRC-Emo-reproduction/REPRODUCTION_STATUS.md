# PRC-Emo reproduction status

本目录是论文 “Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning” 的独立复现目录，不修改 `our-mser` 原有代码。

## 已下载

- 作者 GitHub 仓库的完整代码、脚本、原始数据、论文 PDF 和 README。
- 作者发布的 Hugging Face 检索库原始仓库：`retrieval_repository_source/Emotion_Retrieval_Library.json`。
- 供主流程直接读取的检索库：`data/Emotion_Retrieval_Library_3.json`。

检索库实际包含 36,712 条样本；抽查第一条向量并与代码中的 `all-MiniLM-L6-v2` 对照后，向量维度为 384。Hugging Face README 中的 768 维说明与实际文件不一致，当前应以实际文件和代码为准。

## 已完成检查

- `src` 下 Python 文件通过 `python -m compileall -q src`。
- 原始数据文件齐全：IEMOCAP 和 MELD 的 train/valid/test。
- `reformat_data_ft_llm_combine.py` 的主检索流程读取 `data/Emotion_Retrieval_Library_3.json`，使用 `all-MiniLM-L6-v2`、FAISS L2 距离、top-k=3。
- 训练集检索逻辑会排除当前对话和 utterance，避免检索到当前样本本身。

## 当前还不能直接端到端运行的部分

1. **中间描述文件缺失**：仓库没有提供显式情绪、隐式情绪和说话人特征的生成结果，需要用本地 Qwen2.5 重新生成，并且输出文件名必须匹配后处理代码的读取规则。
2. **模型路径已统一**：`llm_emotion_extract_v2.py`、`llm_bio_extract_v2.py`、`ft_llm_cl.py` 和 `train_llm.sh` 已统一使用 `/home/pc/jcy/models/Qwen2.5-7B-Instruct`。中间描述和说话人特征也约定使用同一个 Qwen2.5 模型。
3. **检索库文件名不一致**：本次正式复现使用作者 Hugging Face 下载的检索库，已经放置为主组合脚本读取的 `data/Emotion_Retrieval_Library_3.json`。`get_rag_final.py` 是作者提供的“根据本地数据重新构建检索库”脚本，它输出另一个 `Emotion_Retrieval_Library.json`，本次不运行它，也不使用它覆盖 Hugging Face 检索库。
4. **依赖声明不完整**：代码实际还需要 `transformers`、`sentence-transformers`、`faiss-cpu` 或 `faiss-gpu`、`numpy`、`tqdm` 等；原始 `requirements.txt` 没有完整列出这些包。
5. **CUDA 被写死**：检索模型在多个脚本中使用 `device='cuda'` 或 `cuda:0`，需要根据本机 GPU/CPU 环境改为可配置参数。
6. **训练入口是 Linux Bash**：`scripts/train_llm.sh` 使用 Bash 语法和 Linux 路径，在 Windows PowerShell 下不能直接运行，需要通过 WSL/Linux 环境运行，或增加 PowerShell/Python 启动器。
7. **额外依赖清单已补充**：`requirements_server_extra.txt` 列出 `trl`、`lightning`、`bitsandbytes`、`sentence-transformers`、`faiss-cpu` 和可选的 `flash-attn`。训练代码支持通过 `PRC_EMO_ATTN_IMPL=sdpa` 绕过 FlashAttention 安装问题。

## 建议的下一步

1. 保留作者原始脚本不动，新增 Qwen2.5 配置/启动脚本。
2. 先只处理 MELD，生成少量样本并打印一条最终 Prompt，验证描述、说话人特征和 3 条检索示例都被正确拼接。
3. 再生成完整 train/valid/test 中间文件，运行一次单 epoch 的 LoRA smoke test。
4. 流程稳定后分别为 IEMOCAP 和 MELD 训练独立 LoRA adapter，并开启课程学习。

## 服务器运行入口

已新增：

- `scripts/generate_prompt_data.py`：只生成包含检索示例的最终 JSONL，不加载训练模型。
- `scripts/run_reproduction_3gpu.sh`：GPU0 顺序生成两类中间特征和最终 Prompt；随后三张卡并行训练 IEMOCAP，完成后再并行训练 MELD。
- `scripts/summarize_results.py`：汇总训练状态、评估指标和失败样本。

评估结果 JSON 中的 `failure_samples` 保存完整的 utterance、messages、Prompt、gold label、预测标签和模型原始回答；`runs/RESULTS.md` 会汇总这些信息。
