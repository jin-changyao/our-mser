# Server Model Setup

Model files are not tracked by git.

Create a shared model directory:

```bash
mkdir -p /home/pc/jcy/models
```

Copy LLaMA2 from the existing SpeechCueLLM location:

```bash
rsync -avP /home/pc/jcy/SpeechCueLLM/LLM_bases/Llama-2-7b-hf/ \
  /home/pc/jcy/models/Llama-2-7b-hf/
```

Copy Qwen2.5 from the existing MSE-Adapter location:

```bash
rsync -avP /home/pc/jcy/MSE-Adapter/models/Qwen2.5-7B-Instruct/ \
  /home/pc/jcy/models/Qwen2.5-7B-Instruct/
```

Check:

```bash
du -sh /home/pc/jcy/models/Llama-2-7b-hf
du -sh /home/pc/jcy/models/Qwen2.5-7B-Instruct
ls -lh /home/pc/jcy/models
```

If disk space is tight, use symlinks instead of copying:

```bash
ln -s /home/pc/jcy/SpeechCueLLM/LLM_bases/Llama-2-7b-hf \
  /home/pc/jcy/models/Llama-2-7b-hf

ln -s /home/pc/jcy/MSE-Adapter/models/Qwen2.5-7B-Instruct \
  /home/pc/jcy/models/Qwen2.5-7B-Instruct
```
