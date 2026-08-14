---
license: cc-by-4.0
---

# 📚 PRC-Emo: A Demonstration Retrieval Repository for Emotion Recognition in Conversation

This repository introduces the first dedicated **demonstration retrieval repository** for Emotion Recognition in Conversation (ERC), as presented in the paper:

> "Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning" (AAAI 2026)

It provides high-quality, utterance-level emotional exemplars that support **retrieval-augmented prompting (RAP)** and significantly improve LLM-based emotional reasoning.

---

## 📊 Dataset Summary

The PRC-Emo repository consists of **36,712** utterance-level samples aggregated from four sources:

| Source | Count |
| :--- | :--- |
| GPT-4o Augmented Emotional Dialogues (Human Verified) | 14,009 |
| IEMOCAP | 5,163 |
| MELD | 9,989 |
| EmoryNLP | 7,551 |
| **Total** | **36,712** |

### Data Fields

Each sample is a complete entry designed for retrieval and training, including the following fields:

* **Utterance text**
* **Emotion label**
* **Source dataset**
* **Dialogue ID**
* **Utterance position**
* **SBERT embedding vector** (768-dim, for retrieval)

The repository is highly suitable for **retrieval-augmented prompting (RAP)**, **few-shot prompting**, and **ERC model training**.

---

## 🛠️ How the Repository Was Built

Due to significant emotion imbalance in existing ERC datasets, we constructed this repository using a multi-stage approach:

### 1. High-Quality Generation with GPT-4o

High-quality emotional dialogues were generated using a two-stage prompting strategy across **six real-world scenarios**: *healthcare, workplace, education, family, social, and entertainment.*

* **Stage 1: Topic Generation**
    * GPT-4o generated 30 diverse subtopics for each scenario.
* **Stage 2: Dialogue Generation**
    * GPT-4o produced two-speaker emotional dialogues with sentence-level emotion labels. **Low-frequency emotions were intentionally reinforced** to ensure balance.

### 2. Label Masking and Rigorous Human Verification

A strict annotation pipeline was used to ensure maximum quality:

1.  GPT-4o produces labeled dialogues.
2.  Emotion labels are automatically masked/removed.
3.  Two human annotators independently **re-label every utterance**.
4.  A sample is kept **only if both annotators exactly match** the original GPT-4o label.

This process was repeated for three rounds to ensure the resulting dataset is diverse, balanced, and of the highest quality.

---

## 💻 Data Format

### Example JSON Entry

```json
{
  "text": "I’ve been feeling great lately! I just had a checkup, and the doctor said everything looks good!",
  "label": "happy",
  "dataset": "my_data",
  "conversation_id": null,
  "utterance_id": 0,
  "vector": [0.1532, -0.9183, ...]
}

{
  "text": "How you doing?",
  "label": "neutral",
  "dataset": "iemocap",
  "conversation_id": "Ses03F_impro06",
  "utterance_id": 0,
  "vector": [0.1532, -0.9183, ...]
}

```
---
## 🎯 Recommended Use Cases

* **Retrieval-Augmented Prompting (RAP)**
    * Retrieve the top-$k$ most similar emotional utterances (using SBERT cosine similarity) and insert them into your LLM prompts as contextual examples.
* **Few-Shot Prompting**
    * Utilize the repository as a high-quality demonstration bank for LLM-based ERC tasks.

---

## 📝 Additional Notes

* **Embeddings:** The provided embeddings were generated using **SBERT** (768-dim). Users may recompute and replace the `embedding` field if a different encoder is preferred.
* **Contact:** 963707605@mail.dlut.edu.cn
* **License:** The dataset is released under **CC BY 4.0**.