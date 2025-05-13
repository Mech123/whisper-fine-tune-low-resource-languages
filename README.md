# 🔊 Breaking Language Barriers: Fine-Tuning Whisper for Bengali and Telugu ASR

This project fine-tunes OpenAI’s Whisper model for **automatic speech recognition (ASR)** in low-resource Indian languages—**Bengali** and **Telugu**—using parameter-efficient fine-tuning (PEFT) methods such as **LoRA**, **BitFit**, and **Adapter Layers**.

---

## 🧠 Core Libraries & Technologies

The following LLM and deep learning libraries were used in this project:

- `openai/whisper` – Pretrained ASR model for multilingual audio
- `transformers` – Model interface & tokenization (Hugging Face)
- `datasets` – Data loading and preprocessing (Hugging Face)
- `peft` – Parameter-efficient fine-tuning (LoRA, BitFit, Adapters)
- `torchaudio` – Audio loading, processing, spectrograms
- `pytorch` – Core deep learning framework for training
- `accelerate` – Easy multi-GPU and mixed-precision training
- `tensorboard` – Real-time training and evaluation tracking

---

## 📌 Project Highlights

- 🔁 Fine-tunes **Whisper Small** using LoRA, BitFit, Adapter Layers
- 📈 Measures Word Error Rate (WER) on Bengali & Telugu ASR tasks
- 🧪 Uses SpecAugment for better generalization
- 🛠️ Includes a CLI pipeline for dataset loading, training, and evaluation
- 🧩 Designed to be modular, extensible, and reproducible

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/whisper-fine-tune-low-resource-languages.git
cd whisper-fine-tune-low-resource-languages
