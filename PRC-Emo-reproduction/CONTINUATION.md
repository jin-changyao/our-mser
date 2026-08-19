# PRC-Emo 复现接续文档

更新时间：2026-08-19
仓库：`https://github.com/jin-changyao/our-mser`
复现目录：`our-mser/PRC-Emo-reproduction`

这份文档是当前复现工作的主接续文档。换电脑后，先执行：

```bash
git clone https://github.com/jin-changyao/our-mser.git
cd our-mser/PRC-Emo-reproduction
```

然后阅读本文，再根据实际服务器的 GPU 编号和模型路径运行脚本。

> **当前接续点（2026-08-19，优先级最高）**
>
> 本文前面和后面的部分保留了早期复现记录。当前实际工作已经进入“清洗说话人特征后，统一重跑五种单 seed 方法”的阶段；如果旧记录与本节冲突，以本节为准。

### 当前目标

使用同一套数据划分、Qwen2.5-7B-Instruct、LoRA/课程学习配置和 `seed=42`，在 IEMOCAP、MELD 上分别比较五种输入条件：

| 方法 | 输入内容 |
|---|---|
| `none` | 对话上下文和目标句，不加入额外特征 |
| `emotion` | 显式情绪描述和隐式情绪描述 |
| `speaker` | 清洗后的说话人描述 |
| `retrieval` | 检索到的相似情绪示例 |
| `full` | 情绪描述、清洗后的说话人描述和检索示例 |

每个数据集只跑一个 `seed=42`，用于先判断不同方法的相对效果。

### 当前已完成

1. 已确认 IEMOCAP 和 MELD 原始 `spdescV6` 都存在污染：模型常常生成多个 `Response`，第一个描述通常有效，后面继续生成与说话人特征无关的任务文本。
2. 已在本地完成说话人特征清洗：保留第一个完整 `Response`，删除后续续写；没有有效 `Response` 的样本保留为空字符串。
3. 已为缺失说话人描述设计固定占位文本：

   ```text
   No reliable speaker description is available.
   ```

   这样不会改变 Prompt 模板结构。
4. 已新增自动化脚本：

   ```text
   PRC-Emo-reproduction/scripts/trim_speaker_features.py
   PRC-Emo-reproduction/scripts/generate_experiment_prompts.py
   PRC-Emo-reproduction/scripts/run_clean_five_methods.sh
   ```

5. 已通过 Python 编译检查和 prompt 构造 smoke test。清洗报告见：

   ```text
   speaker_feature_analysis/TRIM_REPORT.md
   ```

清洗后统计如下：

| 数据集/划分 | 总数 | 有效描述 | 空描述 |
|---|---:|---:|---:|
| IEMOCAP train | 5163 | 5163 | 0 |
| IEMOCAP valid | 647 | 647 | 0 |
| IEMOCAP test | 1623 | 1532 | 91 |
| MELD train | 9989 | 9988 | 1 |
| MELD valid | 1109 | 1109 | 0 |
| MELD test | 2610 | 2610 | 0 |

### Prompt 是否已经生成

最终的五方法 Prompt **还没有作为运行产物提交到 GitHub，也不应提交到 GitHub**。上传清洗后的六个说话人特征 JSON 和两个运行脚本后，在服务器执行自动脚本，才会生成全部 30 个 Prompt 文件：

```text
data/{iemocap,meld}.{train,valid,test}.0shot_w5_experiment_{none,emotion,speaker,retrieval,full}.jsonl
```

Prompt 生成在服务器上串行执行，使用 `PROMPT_GPU`；训练阶段才会按 GPU 波次并行执行。因此，训练不会在 Prompt 尚未完整生成时启动。

### 上传到服务器

本地清洗后的六个 JSON 位于 `speaker_feature_analysis/trimmed/`。它们是运行产物，不属于代码提交内容。上传方式见下方命令中的服务器路径：

```powershell
$LOCAL_ROOT = "C:\Users\Administrator\Desktop\LLM-MER\our-mser-main\github-latest\our-mser"
$SERVER = "pc@10.214.224.230"
$REMOTE_ROOT = "/home/pc/jcy/Our-MSER/PRC-Emo-reproduction"

scp `
  "$LOCAL_ROOT\PRC-Emo-reproduction\scripts\generate_experiment_prompts.py" `
  "$LOCAL_ROOT\PRC-Emo-reproduction\scripts\run_clean_five_methods.sh" `
  "$LOCAL_ROOT\PRC-Emo-reproduction\scripts\trim_speaker_features.py" `
  "${SERVER}:${REMOTE_ROOT}/scripts/"

scp `
  "$LOCAL_ROOT\speaker_feature_analysis\trimmed\iemocap.test_spdescV6_trimmed_Qwen2.5-7B-Instruct.json" `
  "$LOCAL_ROOT\speaker_feature_analysis\trimmed\iemocap.train_spdescV6_trimmed_Qwen2.5-7B-Instruct.json" `
  "$LOCAL_ROOT\speaker_feature_analysis\trimmed\iemocap.valid_spdescV6_trimmed_Qwen2.5-7B-Instruct.json" `
  "$LOCAL_ROOT\speaker_feature_analysis\trimmed\meld.test_spdescV6_trimmed_Qwen2.5-7B-Instruct.json" `
  "$LOCAL_ROOT\speaker_feature_analysis\trimmed\meld.train_spdescV6_trimmed_Qwen2.5-7B-Instruct.json" `
  "$LOCAL_ROOT\speaker_feature_analysis\trimmed\meld.valid_spdescV6_trimmed_Qwen2.5-7B-Instruct.json" `
  "${SERVER}:${REMOTE_ROOT}/data/"
```

### 服务器运行

```bash
cd /home/pc/jcy/Our-MSER/PRC-Emo-reproduction
conda activate speechcuellm
chmod +x scripts/generate_experiment_prompts.py scripts/run_clean_five_methods.sh
bash -n scripts/run_clean_five_methods.sh

PRC_EMO_ATTN_IMPL=sdpa \
GPU0=0 GPU1=1 GPU2=2 PROMPT_GPU=0 \
nohup bash scripts/run_clean_five_methods.sh \
  > logs/clean_five_methods_launcher.log 2>&1 &
echo $!
```

调度顺序为：

1. IEMOCAP 生成五种 Prompt；
2. IEMOCAP 第一波训练 `none`、`emotion`、`retrieval`，分别使用 GPU0、GPU1、GPU2；
3. 第一波全部结束后训练 `speaker`、`full`；
4. IEMOCAP 完成后，对 MELD 重复同样流程。

运行目录为：

```text
runs/clean_five_methods_seed42_<timestamp>/
logs/clean_five_methods_seed42_<timestamp>/
```

最终汇总文件为运行目录下的 `RESULTS.md`，状态表为 `run_status.tsv`。查看总启动日志：

```bash
tail -f logs/clean_five_methods_launcher.log
```

### 当前下一步

1. 将三个脚本和六个清洗后的说话人特征上传服务器；
2. 运行上面的自动化脚本，确认 Prompt 预检查通过；
3. 待五种方法全部完成后，下载 `RESULTS.md`、`run_status.tsv`、训练日志和各方法的最终结果 JSON；
4. 对比两数据集的 weighted F1、macro F1、accuracy 以及失败样本数量；
5. 在单 seed 结果稳定后，再决定是否进行多 seed 复核。

## 1. 我们在复现什么

目标论文是：

> Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning

论文方法的核心是三个输入增强模块：

1. 说话人特征：描述说话人的相对稳定性格和表达特征；
2. 显式/隐式情绪描述：由 LLM 生成当前话语的表面情绪和潜在心理状态描述；
3. 检索增强：从 ERC 检索库中找到相似的带标签情绪表达，加入 Prompt。

在此基础上，作者使用 LoRA 微调和课程学习训练 LLM 进行对话情绪分类。

我们的复现适配使用本地 `Qwen2.5-7B-Instruct`：

```text
原始对话
  → Qwen2.5 生成显式/隐式情绪描述
  → 加入说话人特征
  → 从检索库检索相似样本
  → 构造增强 Prompt
  → Qwen2.5 LoRA 微调
  → IEMOCAP/MELD 情绪分类评估
```

## 2. 当前已经完成的工作

### 2.1 仓库和复现目录

- 已在 `our-mser/PRC-Emo-reproduction` 下建立独立复现目录；
- 没有覆盖原有 `LLM_code`、`feature_utils`、`PROCESSED_DATASET` 等 Our-MSER 代码；
- 作者的基础代码、数据处理代码、检索代码和训练代码已经放入该目录；
- 已统一使用本地 Qwen2.5 模型路径配置；
- 已补充服务器依赖说明和运行说明。

### 2.2 训练兼容性修复

已经处理过的主要兼容性问题：

- Qwen2.5 tokenizer 已经有原生 chat template，不能重复调用 TRL 的 `setup_chat_format`；
- 训练代码现在会保留已有 chat template，只有 tokenizer 没有模板时才使用 TRL fallback；
- 根据当前 SFTTrainer API 调整了数据处理参数，使用 `processing_class`；
- 增加了 SDPA attention fallback，服务器上通常使用：

```bash
PRC_EMO_ATTN_IMPL=sdpa
```

- 固定了服务器端关键依赖版本，见 `requirements_server_extra.txt`；
- `bitsandbytes` 需要在服务器环境中实际安装，否则 Qwen2.5 4-bit 加载会失败；
- 当前服务器曾遇到 Linux kernel 5.4 的 accelerate 警告，该警告不是代码逻辑错误，但可能影响稳定性。

### 2.3 情绪特征生成和修复

`src/llm_emotion_extract_v2.py` 已改为更适合实际运行的版本，主要功能包括：

- 读取 IEMOCAP/MELD 原始对话；
- 调用 Qwen2.5 生成 `surface_emotion` 和 `implicit_emotion`；
- 保存解析后的 JSON 特征；
- 保存每条样本的原始 Prompt、原始模型回答和解析错误；
- 检查已有缓存是否有效；
- 生成失败时返回非零状态；
- 支持 smoke test、强制重新生成和部分样本生成。

另外新增：

```text
scripts/repair_emotion_features.py
```

该工具可以定位 `.raw.jsonl` 中解析失败的样本，只重新生成失败样本，并保留备份和修复记录。

重要说明：Prompt 中虽然有“20–50 words”的要求，但当前解析器不会因为描述少于 20 个词就判定失败。真正的失败条件是空字段、`No prediction` 或无法解析 JSON。此前曾临时去掉词数要求，后来已经恢复原 Prompt 约束，当前代码不会把词数作为硬错误标准。

### 2.4 已完成的特征生成状态

服务器上已经完成 IEMOCAP 和 MELD 的情绪特征生成；IEMOCAP 曾有 3 条失败样本，MELD 曾有 6 条失败样本，之后使用修复脚本进行了补生成。用户已经确认最终特征生成完成。

生成结果通常位于：

```text
data/iemocap.train_ImplicitEmotion_V3_Qwen2.5-7B-Instruct.json
data/iemocap.valid_ImplicitEmotion_V3_Qwen2.5-7B-Instruct.json
data/iemocap.test_ImplicitEmotion_V3_Qwen2.5-7B-Instruct.json
data/meld.train_ImplicitEmotion_V3_Qwen2.5-7B-Instruct.json
data/meld.valid_ImplicitEmotion_V3_Qwen2.5-7B-Instruct.json
data/meld.test_ImplicitEmotion_V3_Qwen2.5-7B-Instruct.json
```

这些生成结果属于运行产物，默认被 `.gitignore` 忽略，不会随 GitHub 仓库同步。换机器时需要重新生成，或者从服务器单独复制。

## 3. 当前已有的主要运行脚本

### 3.1 完整复现流程

```text
scripts/run_reproduction_3gpu.sh
```

作用：检查或生成特征，生成检索 Prompt，然后使用三张 GPU 训练 IEMOCAP 和 MELD。

### 3.2 特征生成后再训练

```text
scripts/run_features_parallel_then_train.sh
```

作用：GPU0/GPU1 并行生成两个数据集的情绪特征，生成完成后再进行训练。已有有效文件时会跳过对应生成步骤。

### 3.3 MELD 单数据集 GPU1 训练

```text
scripts/run_meld_gpu1_after_features.sh
```

作用：不重新生成情绪特征，在 GPU1 上重新构造 MELD 检索 Prompt，然后按 seed 42、43、44 顺序训练 MELD。

默认输出：

```text
runs/meld_gpu1_after_features/
logs/meld_gpu1_after_features/
```

### 3.4 IEMOCAP 四组消融实验

目前新增了两个文件：

```text
scripts/generate_iemocap_ablation_data.py
scripts/run_iemocap_ablation_2gpu.sh
```

四种条件为：

| GPU | 条件 | Prompt 中包含的额外信息 |
|---|---|---|
| GPU0 | `none` | 无说话人特征、无情绪描述、无检索；保留对话上下文 |
| GPU0 | `speaker` | 仅说话人特征 |
| GPU2 | `explicit` | 仅显式情绪描述，不加入隐式情绪和说话人特征 |
| GPU2 | `retrieval` | 仅检索增强，不加入情绪描述和说话人特征 |

四组使用相同的：

- IEMOCAP 数据划分；
- 上下文窗口 `window=5`；
- Qwen2.5-7B-Instruct；
- LoRA 配置；
- 课程学习配置；
- 随机种子，默认 `seed=42`。

该实验一共运行四个任务，不是四种条件各跑三个 seed。若要重复其他 seed，可以修改 `SEED` 后重新运行。

### 3.5 已完成的 IEMOCAP 四组消融实验（seed=42）

截至 2026-08-19，四组 Prompt 生成和训练任务均正常结束（各任务 exit code 为 0）。四组数据规模一致：训练集 5163、验证集 647、测试集 1623。测试结果如下：

| 条件 | weighted F1 | macro F1 | accuracy | 失败样本数 |
|---|---:|---:|---:|---:|
| `none` | 0.6889 | 0.6722 | 0.6882 | 506 |
| `explicit` | **0.7175** | **0.7017** | **0.7166** | 460 |
| `retrieval` | 0.6985 | 0.5917 | 0.6987 | 489 |
| `speaker` | 0.6670 | 0.0408 | 0.6482 | 571 |

相对 `none`，`explicit` 的 weighted F1、macro F1、accuracy 分别提高 0.0286、0.0295、0.0284；`retrieval` 的 weighted F1 和 accuracy 小幅提高，但 macro F1 明显下降；`speaker` 的结果暂时不能用来判断说话人特征是否有害。

这只是单个 seed 的结果，不能作为论文级结论。当前诊断是：

- `explicit` 是目前最清晰、最稳定的正向贡献。日志中的预测只包含六个合法情绪类别，且没有发现同类的格式污染；
- `retrieval` 有 1 个预测被解析成了类似 `sad (from iemocap, distance:` 的异常类别，导致官方 macro F1 被额外类别拉低。这个样本不能在不重跑的情况下直接改写指标；
- `speaker` 的输入特征存在严重污染。检查发现已有 `spdescV6` 文件包含很长的对话、HTML 转义文本和无关任务续写，而不是干净的说话人画像；测试集中至少 91/1623 条预测因此变成了整段 Prompt/HTML 字符串形式的“类别”。所以 `speaker` 的 macro F1=0.0408 主要反映输入/解析失败，不能解释为说话人特征本身的真实效果；
- `speaker` 任务的训练步数也异常偏大（约 1404，而其他条件约 378–540），与说话人 Prompt 过长相互印证，可能触发了截断或样本打包差异。

本次结果和诊断来自服务器运行产物：

```text
runs/iemocap_ablation_2gpu/RESULTS.md
logs/iemocap_ablation_2gpu/train_*.log
logs/iemocap_ablation_2gpu/prompt_*.log
```

## 4. 当前实验状态

### 4.1 IEMOCAP 消融实验：已完成，但需要清理后复跑 `speaker`

本轮已经使用以下命令完成四组 `seed=42` 运行：

```bash
cd /home/pc/jcy/Our-MSER/PRC-Emo-reproduction
conda activate speechcuellm

chmod +x scripts/run_iemocap_ablation_2gpu.sh

PRC_EMO_ATTN_IMPL=sdpa \
GPU0=0 GPU2=2 SEED=42 \
bash scripts/run_iemocap_ablation_2gpu.sh
```

结果位置：

```text
runs/iemocap_ablation_2gpu/RESULTS.md
runs/iemocap_ablation_2gpu/run_status.tsv
runs/iemocap_ablation_2gpu/finetuned_llm/
logs/iemocap_ablation_2gpu/
```

Prompt 中间文件为：

```text
data/iemocap.*.0shot_w5_ablation_none.jsonl
data/iemocap.*.0shot_w5_ablation_speaker.jsonl
data/iemocap.*.0shot_w5_ablation_explicit.jsonl
data/iemocap.*.0shot_w5_ablation_retrieval.jsonl
```

如果怀疑旧 Prompt 文件不完整，可以强制重建：

```bash
FORCE_PROMPT_REGEN=1 \
PRC_EMO_ATTN_IMPL=sdpa \
GPU0=0 GPU2=2 SEED=42 \
bash scripts/run_iemocap_ablation_2gpu.sh
```

当前不能直接把四组排成最终可信的模块贡献排序。下一步应先重新生成并验证干净的说话人特征，再至少复跑 `speaker` 和完整方法；如果修正了输出解析规则，也应在同一规则下复跑所有消融条件。

### 4.2 MELD 正式训练结果：待整理

MELD 的 GPU1 训练脚本已经准备好，但需要检查服务器上实际训练日志和各 seed 的最终指标。不要把之前特征文件为 `No prediction` 时的训练指标当作有效结果。

### 4.3 多 seed 稳定性

当前消融脚本默认每种条件只跑 `seed=42`，适合先判断模块贡献。若需要论文级比较，应对四种条件分别运行多个 seed，并报告 `mean ± std`，不能只比较单次结果。

## 5. 如何查看单个任务结果

以 MELD 的 seed42 为例：

```bash
cd /home/pc/jcy/Our-MSER/PRC-Emo-reproduction

SEED_DIR="runs/meld_gpu1_after_features/finetuned_llm/meld_qwen25_ImplicitEmotion_V3_gpu1_after_feature_repair_seed42"

cat "$SEED_DIR/all_phase_test_results.json"
find "${SEED_DIR}_final_full_finetune" -name "result_*.json" -print
cat "${SEED_DIR}_final_full_finetune"/result_test_final_full_step-*.json
tail -n 100 logs/meld_gpu1_after_features/meld_seed42_gpu1.log
```

详细 `result_*.json` 中包含：

- weighted F1；
- macro F1；
- accuracy；
- 真实标签和预测标签；
- 失败样本数量；
- 失败样本的句子、完整 Prompt、messages 和模型原始输出。

所有任务完成后，脚本才会生成最终的 `RESULTS.md` 汇总文件；但单个 seed 完成后，其 JSON 结果和日志即可查看。

## 6. 服务器环境和路径

当前服务器使用过的路径是：

```text
服务器：pc@10.214.224.230
复现目录：/home/pc/jcy/Our-MSER/PRC-Emo-reproduction
Qwen2.5：/home/pc/jcy/models/Qwen2.5-7B-Instruct
环境：speechcuellm
```

如果换服务器，只需要修改脚本中的：

```bash
PROJECT_ROOT=/your/path/PRC-Emo-reproduction
MODEL_PATH=/your/path/Qwen2.5-7B-Instruct
```

当前主要依赖安装方式：

```bash
python -m pip install -r requirements_server_extra.txt
```

重要依赖版本见 `requirements_server_extra.txt`，尤其注意 `transformers`、`peft`、`trl`、`bitsandbytes` 之间的兼容性。

## 7. GitHub 不会自动包含的运行产物

由于文件太大或属于运行结果，以下内容默认不提交到 GitHub：

- `data/Emotion_Retrieval_Library_3.json`；
- `models/all-MiniLM-L6-v2` 的模型权重；
- Qwen2.5-7B-Instruct 模型本体；
- 情绪特征 JSON；
- 说话人特征 JSON；
- 生成后的 JSONL Prompt；
- `runs/`、`logs/`、模型 checkpoint；
- Python cache 和迁移压缩包。

因此换电脑后，`git clone` 能恢复代码和说明，但仍需要单独准备：

1. Qwen2.5-7B-Instruct；
2. `all-MiniLM-L6-v2`；
3. `Emotion_Retrieval_Library_3.json`；
4. 如果不想重新生成，还需要从服务器复制已经生成的情绪/说话人特征和 Prompt 文件。

## 8. 当前 Git 状态和提交说明

本次更新前，GitHub `main` 的最新提交为：

```text
3118ea6 Document current PRC-Emo reproduction status
```

已提交到仓库的主要修改包括：

- Qwen2.5 chat template 兼容修复；
- 训练数据处理兼容修复；
- 情绪特征生成器重写和失败校验；
- Qwen2.5 生成参数和缓存逻辑调整；
- 并行特征生成、顺序训练脚本；
- 失败情绪特征定点修复工具；
- 服务器依赖和运行说明。

此前已经提交到仓库的复现代码和脚本包括：

```text
PRC-Emo-reproduction/CONTINUATION.md
PRC-Emo-reproduction/scripts/generate_iemocap_ablation_data.py
PRC-Emo-reproduction/scripts/run_iemocap_ablation_2gpu.sh
PRC-Emo-reproduction/scripts/run_meld_gpu1_after_features.sh
```

本次更新只修改 `CONTINUATION.md`，补充 IEMOCAP 消融实验的真实结果和数据污染诊断。

不要提交仓库根目录下的迁移压缩包、stage 临时目录、训练结果和日志。

## 9. 下一步优先级

1. 重新生成并检查干净的 IEMOCAP/MELD 说话人特征，避免把整段对话或无关模型续写写入画像；
2. 明确输出解析策略：合法情绪标签严格按六类统计，异常字符串单独报告，不能混入类别集合；
3. 复跑 IEMOCAP 的 `speaker` 条件和完整方法，确认清理后的真实影响；
4. 检查 MELD GPU1 的三个 seed 结果和失败样本；
5. 对四种消融条件运行多个 seed，并报告 `mean ± std`；
6. 在完成上述复核前，不把当前单 seed 结果写成完整的模块贡献结论。
