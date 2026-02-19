import telebot
from transformers import pipeline

# Замени на свой токен от BotFather
TOKEN = 'ТВОЙ_ТЕЛЕГРАМ_ТОКЕН'
bot = telebot.TeleBot(TOKEN)

print("Загрузка обученной модели...")
# Загружаем модель, которую мы только что обучили
generator = pipeline("text-generation", model="./final_model", tokenizer="./final_model")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Пришли мне данные студента в формате:\nФИО, Балл, Посещаемость, Заметки")

@bot.message_handler(func=lambda message: True)
def generate_response(message):
    prompt = f"<s>{message.text}\nХарактеристика:"
    
    bot.send_chat_action(message.chat.id, 'typing')
    
    # Генерация
    output = generator(prompt, max_new_tokens=150, temperature=0.7, repetition_penalty=1.2)
    full_text = output[0]['generated_text']
    
    # Отрезаем промпт, оставляем только саму характеристику
    char_text = full_text.split("Характеристика:")[-1].strip().replace("</s>", "")
    
    bot.reply_to(message, char_text)

print("Бот запущен!")
bot.polling()
