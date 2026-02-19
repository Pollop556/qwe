import csv
import random

def process_dataset(input_file, output_file):
    rows = []
    errors = 0
    
    print(f"--- Начало обработки файла: {input_file} ---")
    
    try:
        with open(input_file, mode='r', encoding='utf-8') as f:
            # Читаем заголовок
            header = f.readline().strip()
            
            # Читаем остальные строки через csv reader для проверки формата
            reader = csv.reader(f)
            for i, row in enumerate(reader, start=2):
                # Проверка: в строке должно быть ровно 2 колонки (input и target)
                if len(row) != 2:
                    print(f"Ошибка в строке {i}: ожидалось 2 колонки, найдено {len(row)}")
                    print(f"Содержимое: {row}")
                    errors += 1
                    continue
                
                # Проверка на пустые значения
                if not row[0].strip() or not row[1].strip():
                    print(f"Предупреждение в строке {i}: обнаружена пустая колонка.")
                    errors += 1
                    continue
                
                rows.append(row)
        
        if errors > 0:
            print(f"\nНайдено ошибок в формате: {errors}. Исправьте их перед обучением.")
        else:
            print("Синтаксических ошибок не обнаружено!")

        # Перемешиваем строки (random.shuffle работает на месте)
        random.shuffle(rows)
        
        # Записываем результат
        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL) # Добавляем кавычки для безопасности
            f.write(header + '\n') # Пишем оригинальный заголовок
            writer.writerows(rows)
            
        print(f"--- Успешно! Перемешано строк: {len(rows)} ---")
        print(f"Файл сохранен как: {output_file}")

    except Exception as e:
        print(f"Произошла фатальная ошибка: {e}")

# Запуск скрипта
# Убедись, что твой файл называется dataset.csv или измени имя ниже
process_dataset('dataset.csv', 'shuffled_dataset.csv')
