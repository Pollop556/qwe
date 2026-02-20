import torch
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments

def train():
    # 1. 🚀 Конфигурация модели
    # Используем Qwen 2.5 3B Instruct — лучшая модель для русского языка в этом классе
    model_name = "Qwen/Qwen2.5-3B-Instruct" 
    
    # Максимальная длина контекста. 2048 токенов достаточно для одной характеристики (обычно ~500-1000 слов).
    # Если характеристики очень длинные, можно увеличить до 4096, но это займет больше памяти.
    max_seq_length = 2048 
    
    dtype = None # Автоматическое определение (float16 для T4)
    load_in_4bit = True # Обязательно True для T4 (16GB), иначе модель не влезет или будет медленной

    print(f"🚀 Загрузка модели {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 2. ⚡ Настройка LoRA адаптеров (оптимизация под Qwen)
    # Qwen имеет специфичные модули (q,k,v,o,gate,up,down), мы обучаем их все для лучшего качества.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Ранг адаптера. 16 — золотая середина (можно 32 или 64, но 16 быстрее и памяти меньше)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16, # Alpha = r (стандартная практика)
        lora_dropout = 0, # 0 для скорости
        bias = "none",    # "none" для скорости и памяти
        use_gradient_checkpointing = "unsloth", # Критично для экономии VRAM
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. 📂 Подготовка датасета
    print("📂 Загрузка и подготовка датасета...")
    df = pd.read_csv('shuffled_dataset.csv')
    dataset = Dataset.from_pandas(df)

    # Форматирование под Qwen Chat Template
    def formatting_prompts_func(examples):
        texts = []
        for input_text, output_text in zip(examples["input"], examples["target"]):
            messages = [
                # Системный промпт — задает роль и стиль.
                {"role": "system", "content": "Ты — профессиональный педагог-куратор. Твоя задача — составлять подробные, объективные и педагогически грамотные характеристики на студентов. Стиль изложения: официально-деловой, сдержанный, но содержательный."},
                
                # Входные данные от пользователя
                {"role": "user", "content": f"Составь характеристику на студента по следующим данным: {input_text}"},
                
                # Эталонный ответ (то, чему учим модель)
                {"role": "assistant", "content": output_text}
            ]
            
            # apply_chat_template сам добавит специальные токены <|im_start|> и <|im_end|> для Qwen
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 4. 🔥 Гиперпараметры обучения (Training Arguments)
    # Оптимизировано под dataset ~500 примеров и модель 3B
    print("🔥 Начинаем обучение...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 2,   # Батч 2 на карту (влезает в 16GB)
            gradient_accumulation_steps = 4,   # Эффективный батч = 2 * 4 = 8
            warmup_steps = 10,                 # Разогрев (чуть больше для стабилизации)
            num_train_epochs = 3,              # 3 эпохи обычно идеально для ~500 примеров. 
                                               # Если 1 эпоха, модель недоучится. Если 10 — переучится (зазубрит).
            learning_rate = 2e-4,              # Стандартный LR для QLoRA
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",              # 8-битный оптимизатор экономит память
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Отключаем wandb чтобы не логиниться лишний раз
        ),
    )

    trainer.train()

    # 5. 💾 Сохранение и экспорт
    print("✅ Обучение завершено!")
    
    # Сохраняем адаптеры
    model.save_pretrained("lora_model") 
    tokenizer.save_pretrained("lora_model")

    # Экспорт в GGUF для запуска на ноутбуке
    # Используем q4_k_m — лучший баланс скорости и качества для 3B моделей
    print("📦 Экспорт в GGUF (q4_k_m)...")
    try:
        model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")
        print("🎉 УСПЕХ! Файл GGUF сохранен в папку 'model_gguf'. Скачай его!")
    except Exception as e:
        print(f"❌ Ошибка экспорта GGUF: {e}")

if __name__ == "__main__":
    train()
