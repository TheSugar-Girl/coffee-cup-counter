# ☕ Coffee Cup Counter

Веб-приложение для автоматического подсчёта стаканов и кружек в кофейне с помощью предобученной нейронной сети YOLOv8.

## 🛠️ Технологии

- [YOLOv8](https://docs.ultralytics.com) — детектирование объектов
- [Streamlit](https://streamlit.io) — веб-интерфейс
- [OpenCV](https://opencv.org) — обработка изображений
- [fpdf2](https://py-pdf.github.io/fpdf2) — генерация PDF-отчётов

## 🚀 Установка и запуск

**1. Клонируй репозиторий:**
```bash
git clone https://github.com/TheSugar-Girl/coffee-cup-counter
cd coffee-cup-counter
```

**2. Создай виртуальное окружение:**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Установи зависимости:**
```bash
pip install -r requirements.txt
```

**4. Запусти приложение:**
```bash
streamlit run app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`

## 📋 Возможности

- 📷 Загрузка изображений (JPG, PNG, WEBP)
- 🎥 Обработка видеофайлов
- 📸 Снимок с веб-камеры
- 📊 История запросов в JSON
- 📥 Генерация PDF-отчёта
