import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train_model():
    # 1. Настройка модели и токенизатора
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Добавляем специальные токены, чтобы модель понимала, где вход, а где ответ
    special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 2. Подготовка данных
    # Мы преобразуем CSV в текстовый формат, который понимает GPT
    def prepare_data(file_path):
        df = pd.read_csv(file_path)
        with open("train_text.txt", "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                # Формат: <bos> Ввод <sep> Результат <eos>
                combined_text = f"<bos>{row['input']}\nХарактеристика: {row['target']}<eos>\n"
                f.write(combined_text)

    print("Подготовка данных...")
    prepare_data('shuffled_dataset.csv')

    # 3. Создание датасета для обучения
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="train_text.txt",
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Настройка параметров обучения
    training_args = TrainingArguments(
        output_dir="./results",          # папка для промежуточных данных
        overwrite_output_dir=True,
        num_train_epochs=10,             # количество проходов по датасету
        per_device_train_batch_size=4,   # размер батча (зависит от памяти видеокарты)
        save_steps=500,                  # сохранять чекпоинт каждые 500 шагов
        save_total_limit=2,              # хранить только 2 последних чекпоинта
        learning_rate=5e-5,              # скорость обучения
        warmup_steps=100,                # плавный вход в обучение
        logging_dir='./logs',            # папка для логов
    )

    # 5. Запуск обучения
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("Начало обучения... Это может занять время.")
    trainer.train()

    # 6. Сохранение финальной модели
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Обучение завершено! Модель сохранена в папку 'final_model'.")

if __name__ == "__main__":
    train_model()
