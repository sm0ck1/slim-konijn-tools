# Slim Konijn Tools

FastAPI сервис для анализа слов, предложений и получения синонимов на английском и нидерландском языках.

## Функциональность

- **Анализ слов**: POS-тегирование и лемматизация
- **Анализ предложений**: Грамматическая валидация с dependency parsing
- **Синонимы**: Получение синонимов для английских (WordNet) и нидерландских (FastText) слов
- **Проверка орфографии**: Автоматическая проверка валидности слов через FastText
- **Spell correction**: Поиск похожих слов если слово написано неправильно
- **Автозагрузка моделей**: FastText модели скачиваются автоматически при первом запуске

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. При первом запуске автоматически загрузятся:
   - Stanza модели для английского и нидерландского
   - FastText модели для обоих языков (~6.8GB каждая)
   
**Примечание**: Первый запуск займет время из-за загрузки FastText моделей (общий размер ~13.6GB)

## Запуск

```bash
python main.py
```

Сервис будет доступен на `http://localhost:8000`

Документация API: `http://localhost:8000/docs`

## API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Анализ слова
```bash
POST /analyze/word
{
  "word": "running",
  "language": "en"
}
```

**Ответ:**
```json
{
  "word": "running",
  "pos_tag": "VERB",
  "lemma": "run"
}
```

### 3. Анализ предложения
```bash
POST /analyze/sentence
{
  "sentence": "The cat sits on the mat",
  "language": "en",
  "include_synonyms": true,
  "synonyms_limit": 5
}
```

**Параметры:**
- `include_synonyms` (по умолчанию `true`) - включить синонимы для каждого слова
- `synonyms_limit` (по умолчанию `5`) - максимальное количество синонимов на слово

**Ответ:**
```json
{
  "sentence": "The cat sits on the mat",
  "is_valid": true,
  "words": [
    {
      "word": "The",
      "pos_tag": "DET",
      "lemma": "the",
      "synonyms": [],
      "is_valid": true,
      "suggestions": []
    },
    {
      "word": "cat",
      "pos_tag": "NOUN",
      "lemma": "cat",
      "synonyms": ["feline", "kitty", "kitten", "tomcat", "tabby"],
      "is_valid": true,
      "suggestions": []
    },
    ...
  ],
  "tokens": ["The", "cat", "sits", "on", "the", "mat"]
}
```

**Пример с ошибкой в слове:**
```json
{
  "sentence": "The catt sits",
  "is_valid": true,
  "words": [
    {
      "word": "catt",
      "pos_tag": "NOUN",
      "lemma": "catt",
      "synonyms": [],
      "is_valid": false,
      "suggestions": ["cat", "cats", "cattle", "catty", "matt"]
    }
  ]
}
```

### 4. Пакетный анализ
```bash
POST /analyze/batch
{
  "sentences": ["Hello world", "How are you"],
  "language": "en",
  "include_synonyms": true,
  "synonyms_limit": 5
}
```

**Параметры такие же как в `/analyze/sentence`**

### 5. Получение синонимов
```bash
POST /synonyms
{
  "word": "happy",
  "language": "en",
  "topn": 10
}
```

**Ответ:**
```json
{
  "word": "happy",
  "lemma": "happy",
  "synonyms": ["glad", "joyful", "cheerful", "pleased", "content"],
  "source": "WordNet"
}
```

**Для нидерландского:**
```bash
POST /synonyms
{
  "word": "huis",
  "language": "nl",
  "topn": 10
}
```

**Ответ:**
```json
{
  "word": "huis",
  "lemma": "huis",
  "synonyms": ["woning", "appartement", "gebouw", ...],
  "source": "FastText"
}
```

## Примеры использования

### Python
```python
import requests

# Получить синонимы
response = requests.post('http://localhost:8000/synonyms', json={
    'word': 'happy',
    'language': 'en'
})
print(response.json())

# Анализ предложения с синонимами
response = requests.post('http://localhost:8000/analyze/sentence', json={
    'sentence': 'I am learning Dutch',
    'language': 'en',
    'include_synonyms': True,
    'synonyms_limit': 5
})
print(response.json())

# Анализ предложения без синонимов (быстрее)
response = requests.post('http://localhost:8000/analyze/sentence', json={
    'sentence': 'I am learning Dutch',
    'language': 'en',
    'include_synonyms': False
})
print(response.json())
```

### cURL
```bash
# Английские синонимы
curl -X POST "http://localhost:8000/synonyms" \
  -H "Content-Type: application/json" \
  -d '{"word": "beautiful", "language": "en"}'

# Нидерландские синонимы (требуется FastText модель)
curl -X POST "http://localhost:8000/synonyms" \
  -H "Content-Type: application/json" \
  -d '{"word": "mooi", "language": "nl", "topn": 5}'
```

## Источники данных

- **Английские синонимы**: NLTK WordNet
- **Нидерландские синонимы**: FastText (cc.nl.300.bin - 6.8GB)
- **Проверка орфографии**: FastText для обоих языков (cc.en.300.bin + cc.nl.300.bin)
- **Лемматизация и POS**: Stanza NLP

## Особенности

- ✅ **Автозагрузка моделей**: FastText модели скачиваются автоматически при первом запуске
- ✅ **Проверка слов**: Каждое слово проверяется на валидность через FastText словарь
- ✅ **Spell correction**: Автоматический поиск похожих слов для опечаток
- ✅ **Поддержка двух языков**: Английский и нидерландский
- ⚠️ **Большой размер**: Первая загрузка требует ~13.6GB для обеих FastText моделей
