import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train():
    # 1. Загрузка модели и токенизатора
    # Используем sberbank-ai/rugpt3small_based_on_gpt2, она лучше всего подходит для таких задач
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 2. Подготовка данных
    # Мы склеиваем input и target специальным разделителем <|endoftext|>, 
    # чтобы модель понимала, где заканчиваются данные и начинается характеристика
    df = pd.read_csv("dataset.csv")
    train_text = ""
    for _, row in df.iterrows():
        train_text += f"{row['input']} {row['target']} {tokenizer.eos_token}\n"
    
    with open("train_data.txt", "w", encoding="utf-8") as f:
        f.write(train_text)

    # Создание датасета
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="train_data.txt",
        block_size=256 # Оптимально для характеристик средней длины
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. Улучшенные параметры обучения
    training_args = TrainingArguments(
        output_dir="./rugpt3_student_model",
        overwrite_output_dir=True,
        num_train_epochs=15,          # Увеличили до 15 для глубокого усвоения 167 примеров
        per_device_train_batch_size=4, # Оптимально для памяти Colab
        save_steps=500,
        save_total_limit=2,
        
        # Продвинутые параметры:
        learning_rate=5e-5,           # Чуть ниже стандартного для мягкого дообучения
        weight_decay=0.01,            # Регуляризация, чтобы модель не "зубрила" ФИО
        warmup_steps=100,             # Плавный вход в обучение
        lr_scheduler_type="cosine",   # Постепенное снижение скорости обучения к концу
        logging_steps=10,
        fp16=True,                    # Включаем ускорение на GPU (в Colab должно быть активно)
    )

    # 4. Запуск тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Начинаю обучение на 167 примерах...")
    trainer.train()
    
    # Сохранение финальной версии
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Обучение завершено! Модель сохранена в папку final_model")

if __name__ == "__main__":
    train()
