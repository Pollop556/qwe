import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

def train():
    model_name = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct"
    
    # 1. Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Важно для Llama

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 # Оптимально для T4 GPU в Colab
    )

    # 2. Подготовка датасета (Идеальный промпт)
    df = pd.read_csv('shuffled_dataset.csv')

    def format_chat(example):
        # Используем официальный формат Llama-3 для инструкций
        messages = [
            {"role": "system", "content": "Ты — эксперт-куратор. Пишешь профессиональные характеристики студентов по заданным параметрам. Используй официально-деловой стиль."},
            {"role": "user", "content": f"Сформируй характеристику: {example['input']}"},
            {"role": "assistant", "content": example['target']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    dataset = Dataset.from_pandas(df).map(format_chat)

    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=384, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_func, batched=True, remove_columns=dataset.column_names)

    # 3. Идеальные параметры обучения для 1B модели
    training_args = TrainingArguments(
        output_dir="./vikhr_results",
        num_train_epochs=3,              # 3 эпохи — "золотой стандарт"
        per_device_train_batch_size=2,   # Чтобы не вылететь по памяти
        gradient_accumulation_steps=8,   # Эффективный батч = 16 (2*8)
        learning_rate=2e-5,              # Мягкое обучение
        lr_scheduler_type="cosine",      # Плавное затухание
        warmup_ratio=0.1,                # 10% времени на разогрев
        weight_decay=0.05,               # Профилактика переобучения
        logging_steps=5,
        save_strategy="no",
        fp16=True,                       # Ускорение на GPU
        gradient_checkpointing=True,     # Максимальная экономия VRAM
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("🚀 Запуск идеального обучения Vikhr-1B...")
    trainer.train()
    
    # Сохранение
    model.save_pretrained("./final_vikhr_model")
    tokenizer.save_pretrained("./final_vikhr_model")
    print("✅ Готово! Модель в папке ./final_vikhr_model")

if __name__ == "__main__":
    train()
