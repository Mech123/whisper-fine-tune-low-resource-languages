import os
import argparse
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass
import torch
import evaluate
from typing import Any, Dict, List, Union
import numpy as np
from datasets import Audio
import pickle
from tqdm import tqdm

SAVE_PATH = "evaluation_progress.pkl"
wer = evaluate.load("wer")


def save_progress(progress):
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(progress, f)

def load_progress():
    try:
        with open(SAVE_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def custom_compute_metrics(dataset, model, tokenizer, data_collator, device="cuda", batch_size=8, resume=True):
    from torch.utils.data import DataLoader

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    progress = load_progress() if resume else {"predictions": [], "references": [], "processed_batches": 0}
    predictions, references = progress.get("predictions", []), progress.get("references", [])
    start_batch = progress.get("processed_batches", 0)

    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        if batch_idx < start_batch:
            continue

        input_features = batch["input_features"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            generated_tokens = model.generate(input_features=input_features, language=tokenizer.language, task='transcribe')

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        predictions.extend(decoded_preds)
        references.extend(decoded_labels)

        if (batch_idx + 1) % 20 == 0:
            save_progress({
                "predictions": predictions,
                "references": references,
                "processed_batches": batch_idx + 1
            })

    wer_score = wer.compute(predictions=predictions, references=references) * 100
    print(f"WER: {wer_score}")
    return {"wer": wer_score}


def load_and_prepare_dataset(dataset_name, train_frac, test_frac):
    full_train = load_dataset(dataset_name, split="train")
    full_test = load_dataset(dataset_name, split="test")

    new_train_size = int(train_frac * len(full_train))
    new_test_size = int(test_frac * len(full_test))

    train_sample = full_train.shuffle(seed=42).select(range(new_train_size))
    test_sample = full_test.shuffle(seed=42).select(range(new_test_size))

    return DatasetDict({"train": train_sample, "test": test_sample})


#------------------Custom preprocessing function for dataset-----------------------------
def preprocess_dataset(dataset, feature_extractor, tokenizer):
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad({"input_ids": label_features}, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset path")
    parser.add_argument("--language", type=str, required=True, help="Language for fine-tuning")
    parser.add_argument("--username", type=str, required=True, help="Your Hugging Face username")
    parser.add_argument("--output_dir", type=str, default="./whisper-finetuned")
    parser.add_argument("--train_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.4)
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = f"openai/whisper-{args.model_size}"
    processor = WhisperProcessor.from_pretrained(model_id, language=args.language, task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model = model.to(device)

    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["k_proj", "v_proj", "q_proj", "out_proj"], lora_dropout=0.05, bias="none")
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    dataset = load_and_prepare_dataset(args.dataset, args.train_frac, args.test_frac)
    dataset = preprocess_dataset(dataset, feature_extractor, tokenizer)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    print("\nRunning custom evaluation BEFORE training...")
    # custom_compute_metrics(dataset["test"], model, tokenizer, device=device)
    custom_compute_metrics(dataset["test"], model, tokenizer, data_collator=data_collator, device=device)


    training_args = Seq2SeqTrainingArguments(
        output_dir="lora-checkpoints",
        per_device_train_batch_size=2,  # Reduced from 8 to 2
        per_device_eval_batch_size=2,  # Reduced from 8 to 2
        gradient_accumulation_steps=2,  # Helps with small batch size
        # learning_rate=2e-5, (High)
        learning_rate=1e-5,  #(Medium)
        # learning_rate=5e-6,  #(Low)
        warmup_steps=50,
        num_train_epochs=2,
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=142,
        # save_strategy="epoch",
        save_strategy="steps",
        save_steps=284, 
        save_total_limit=2,
        load_best_model_at_end=True,  # Disabling to save memory
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Mixed precision enabled
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,
        label_names=["labels"],

        # Add AdamW optimizer and related arguments
        optim="adamw_torch",  # AdamW optimizer from PyTorch
        weight_decay=0.01,  # Regularization to prevent overfitting
        adam_epsilon=1e-8,   # Epsilon for numerical stability
        max_grad_norm=1.0,   # Gradient clipping
        # TensorBoard logging
        logging_dir="./lora-tensorboard",  # Directory for TensorBoard logs
        report_to="tensorboard",  # Enables logging to TensorBoard
    )


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # ----------------------------Reload model from saved directory with LoRA---------------------------
    print("\nReloading trained model with LoRA adapters from saved directory...")
    base_model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{args.model_size}")
    loaded_model = get_peft_model(base_model, lora_config)
    loaded_model = PeftModel.from_pretrained(loaded_model, args.output_dir)
    loaded_model = loaded_model.to(device)

    print("\nRunning custom evaluation AFTER training...")
    custom_compute_metrics(dataset["test"], loaded_model, tokenizer, device=device)

    # Push model to Hugging Face Hub
    model_repo = f"{args.username}/whisper-{args.model_size}-{args.language}-finetuned"
    print(f"\nPushing model to Hugging Face Hub at {model_repo}...")
    loaded_model.push_to_hub(model_repo)
    processor.push_to_hub(model_repo)

    print(f"\n✅ Successfully fine-tuned Whisper-{args.model_size} model to {args.language} dataset.")
    print(f"📍 Find your fine-tuned model here: https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
