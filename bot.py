import telebot
from llama_cpp import Llama

# 1. Настройка
# Замени на свой токен от BotFather
TOKEN = 'ТВОЙ_ТЕЛЕГРАМ_ТОКЕН'
# Путь к GGUF файлу, который ты скачал из Colab.
# Например: "model_gguf/model-unsloth.Q4_K_M.gguf"
MODEL_PATH = "model_gguf/model-unsloth.Q4_K_M.gguf" 

bot = telebot.TeleBot(TOKEN)

print(f"Загрузка модели из {MODEL_PATH}...")
# Загружаем GGUF модель. n_gpu_layers=-1 попытается использовать видеокарту, если есть.
# Если нет, будет использовать процессор.
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,           # Контекстное окно
        n_threads=4,          # Количество ядер процессора (если тормозит, уменьши)
        n_gpu_layers=-1       # -1 = все слои на GPU (если есть CUDA)
    )
    print("✅ Модель загружена!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели (проверь путь к файлу!): {e}")
    exit()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я эксперт-педагог. Пришли мне данные студента, и я напишу характеристику.\nПример: Иванов Иван, 4 курс, отличник, активный участник КВН.")

@bot.message_handler(func=lambda message: True)
def generate_response(message):
    user_input = message.text
    
    bot.send_chat_action(message.chat.id, 'typing')

    # Формируем промпт в формате ChatML (для Qwen/Llama-3)
    # Важно соблюдать формат, на котором училась модель!
    messages = [
        {"role": "system", "content": "Ты — профессиональный педагог-куратор. Твоя задача — составлять полные, грамотные и объективные характеристики на студентов для официальных документов. Используй официально-деловой стиль."},
        {"role": "user", "content": f"Напиши характеристику на студента по следующим данным: {user_input}"}
    ]

    # Создаем промпт из истории сообщений
    # Llama-cpp-python имеет встроенный метод для этого, но можно и вручную
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,      # Максимальная длина ответа
        temperature=0.7,     # Креативность (0.7 - сбалансировано)
        top_p=0.9,
    )
    
    # Достаем текст
    text = response['choices'][0]['message']['content']
    
    bot.reply_to(message, text)

print("Бот запущен!")
bot.polling()
