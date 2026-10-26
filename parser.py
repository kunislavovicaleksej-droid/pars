import time
import json
import os
import re
import requests
import feedparser
from bs4 import BeautifulSoup
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()  # подхватывает .env при локальном запуске
except ImportError:
    pass

# ================= НАСТРОЙКИ =================
BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

GROQ_API_KEYS = [k.strip() for k in os.environ.get('GROQ_API_KEYS', '').split(',') if k.strip()]

if not BOT_TOKEN or not GROQ_API_KEYS:
    raise SystemExit(
        "❌ Не заданы переменные окружения BOT_TOKEN и/или GROQ_API_KEYS."
    )

# Желаемый приоритет моделей Groq
PREFERRED_MODELS = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'openai/gpt-oss-20b',
    'openai/gpt-oss-120b',
    'qwen/qwen3-32b'
]

current_key_index = 0

def get_groq_client():
    return OpenAI(
        api_key=GROQ_API_KEYS[current_key_index],
        base_url="https://api.groq.com/openai/v1"
    )

client = get_groq_client()

def get_available_models_for_current_client():
    """Динамически получает список доступных моделей для текущего ключа."""
    global client
    try:
        models_resp = client.models.list()
        available_ids = {m.id for m in models_resp.data}
        filtered = [m for m in PREFERRED_MODELS if m in available_ids]
        if filtered:
            return filtered
    except Exception as e:
        print(f"⚠️ Не удалось определить модели для ключа №{current_key_index + 1}: {e}")
    return PREFERRED_MODELS

# Список RSS-лент
RSS_URLS = [
    'https://www.expats.cz/feed',
    'https://prazska.drbna.cz/rss/',
    'https://www.aktualne.cz/rss/',
    'https://praguedaily.news/feed/',
    'https://czechia-online.cz/feed/',
    'https://ukraina.radio.cz/rss',
    'https://tn.nova.cz/rss',
    'https://plzensky.denik.cz/rss/',
    'https://www.blesk.cz/rss',
    'https://ct24.ceskatelevize.cz/rss/hlavni-zpravy',
    'https://echo24.cz/rss',
    'https://www.novinky.cz/rss',
    'https://420on.cz/rss',
    'https://www.seznamzpravy.cz/rss',
    'https://servis.idnes.cz/rss.aspx?c=zpravodaj',
    'https://cnn.iprima.cz/rss',
    'https://tydenikpolicie.cz/feed/',
    'https://krimi-plzen.cz/feed/',
    'https://brnenska.drbna.cz/rss/',
    'https://www.extra.cz/rss'
]

HISTORY_FILE = 'history.json'
TOPICS_FILE = 'posted_topics.json'
CHECK_INTERVAL = 300  # 5 минут

FIRST_RUN = True

SYSTEM_PROMPT = """Ти — топовий копірайтер із 15-річним стажем, який спеціалізується на створенні контенту для найбільших українських Telegram-каналів у стилі "Труха". 
Твоє завдання — ПЕРЕКЛАСТИ отриманий текст (найчастіше чеський) на українську мову та переробити (зробити рерайт) під формат коротких, клікабельних та динамічних новин.

Працюй за такими суворими правилами:

1. СТРУКТУРА ПОСТУ:
   - Заголовок: Головна суть новини одним динамічним реченням. Наприкінці заголовка ОБОВ'ЯЗКОВО вказуй автора або джерело інформації через тире та кому. Формат: <b>[Суть новини], — [Хто сказав / Джерело]</b>.
   - Відступ (порожній рядок).
   - Опис: Короткий, місткий рерайт решти інформації (1-2 речення). Текст має бути "живим", легким для читання, у стилістиці топових ЗМІ.

2. СУВОРЕ ОБМЕЖЕННЯ НА ФАКТИ (ЖОДНОЇ СЕБЕСТЯБАТИНИ):
   - Використовуй ТІЛЬКИ ту інформацію, яку я тобі скидаю.
   - НЕ додавай нічого від себе, не шукай інформацію в інтернеті, не додумуй контекст, не додавай старі чи супутні факти. Якщо чогось немає в моєму тексті — цього не повинно бути в твоєму результаті.
   - Спікер/автор цитати ОБОВ'ЯЗКОВО має бути в заголовку, як вказано у структурі. Якщо в тексті немає чіткого автора, пиши джерело.

3. ТЕХНІЧНЕ ПРАВИЛО:
   - Відповідай ТІЛЬКИ УКРАЇНСЬКОЮ мовою.
   - Виділяй заголовок жирним шрифтом, використовуючи ТІЛЬКИ HTML-теги <b> та </b>. Не використовуй зірочки (**) для форматування!
"""

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def call_groq_api(system_msg, user_msg):
    global current_key_index, client

    active_models = get_available_models_for_current_client()

    for model_name in active_models:
        attempts = 0
        max_attempts = len(GROQ_API_KEYS)

        while attempts < max_attempts:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e).lower()
                print(f"🔎 RAW ERROR ({model_name}): {e}")

                if "401" in error_str or "403" in error_str or "invalid_api_key" in error_str:
                    print(f"❌ Ключ №{current_key_index + 1} недействителен. Переключаюсь на следующий...")
                    current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                    client = get_groq_client()
                    attempts += 1
                    continue

                elif "429" in error_str or "rate_limit" in error_str:
                    match = re.search(r"try again in (\d+(\.\d+)?)s", error_str)
                    sleep_time = float(match.group(1)) + 0.5 if match else 4.0
                    
                    print(f"⚠️ Лимит (429) на ключе №{current_key_index + 1}. Жду {sleep_time:.1f} сек...")
                    time.sleep(sleep_time)
                    
                    current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                    client = get_groq_client()
                    attempts += 1
                    continue

                elif "decommissioned" in error_str or "model_not_found" in error_str:
                    print(f"⚠️ Модель {model_name} недоступна. Пробую следующую модель...")
                    break

                else:
                    print(f"❌ Ошибка обращения к Groq API ({model_name}): {e}")
                    break

    print("❌ Ни одна из моделей Groq не сработала!")
    return None

def is_duplicate_news(article_text, posted_topics):
    if not posted_topics:
        return False

    recent_topics = posted_topics[-15:]
    topics_list_str = "\n".join([f"- {topic}" for topic in recent_topics])
    
    dup_prompt_system = (
        "Ти — головний редактор. Твоє завдання: перевірити, чи є нова стаття точним дублікатом однієї з подій у списку.\n\n"
        "СУВОРІ ПРАВИЛА:\n"
        "1. Дублікат — це виключно та САМА подія.\n"
        "2. Збіг імен або міст — це НОВА новина!\n"
        "3. Якщо ти хоч трохи сумніваєшся — завжди обирай НОВА.\n\n"
        "Відповідай СУВОРО одним словом:\n"
        "ДУБЛІКАТ\n"
        "НОВА"
    )
    
    dup_prompt_user = f"ОПУБЛІКОВАНІ ТЕМИ:\n{topics_list_str}\n\nТЕКСТ НОВОЇ СТАТТІ:\n{article_text[:800]}"

    res = call_groq_api(dup_prompt_system, dup_prompt_user)
    
    if res and "ДУБЛІКАТ" in res.upper() and len(res) < 15:
        return True
    return False

def get_article_data(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        image_url = None
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            image_url = og_image['content']
            
        return text[:3000], image_url
    except Exception as e:
        print(f"Ошибка при парсинге статьи {url}: {e}")
        return "", None

def send_telegram_message(text, image_url=None):
    if image_url:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        payload = {
            'chat_id': CHAT_ID,
            'photo': image_url,
            'caption': text,
            'parse_mode': 'HTML'
        }
    else:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': text,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code != 200 and image_url:
            print(f"Телеграм не принял картинку, отправляю просто текст...")
            send_telegram_message(text, image_url=None)
        elif response.status_code != 200:
            print(f"Ошибка отправки в ТГ: {response.text}")
    except Exception as e:
        print(f"Ошибка соединения с ТГ: {e}")

# ================= ОСНОВНОЙ ЦИКЛ =================

def check_news():
    global FIRST_RUN
    history = load_json(HISTORY_FILE)
    posted_topics = load_json(TOPICS_FILE)
    new_posts_found = False

    is_first_run = FIRST_RUN and len(history) == 0

    start_time = time.time()
    MAX_RUN_TIME = 800  

    for rss_url in RSS_URLS:
        if (time.time() - start_time) > MAX_RUN_TIME:
            print(f"⏱️ Превышено максимальное время выполнения ({MAX_RUN_TIME}с). Сохраняю прогресс...")
            break

        if not is_first_run:
            print(f"Проверяю ленту: {rss_url}")
            
        try:
            resp = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            feed = feedparser.parse(resp.content)
            
            for entry in feed.entries[:3]:
                if (time.time() - start_time) > MAX_RUN_TIME:
                    break

                link = getattr(entry, 'link', None)
                title = getattr(entry, 'title', 'Без заголовка')
                
                if link and link not in history:
                    if is_first_run:
                        print(f"[Старая новость, пропускаю] {title}")
                        history.append(link)
                        new_posts_found = True
                    else:
                        print(f"🔥 Найден новый пост! Читаю статью: {link}")
                        
                        article_text, image_url = get_article_data(link)
                        
                        if len(article_text) > 200: 
                            if is_duplicate_news(article_text, posted_topics):
                                print(f"🙈 Нейросеть определила, что это ДУБЛИКАТ. Пропускаю...")
                                history.append(link)
                                new_posts_found = True
                                continue
                            
                            ai_post = call_groq_api(SYSTEM_PROMPT, article_text)
                            
                            if ai_post:
                                final_message = f"{ai_post}\n\nПост взял на сайте:\n{link}"
                                send_telegram_message(final_message, image_url)
                                print("Пост успешно переписан и отправлен в ТГ!")
                                
                                first_line = ai_post.split('\n')[0].replace('<b>', '').replace('</b>', '')
                                posted_topics.append(first_line)
                                save_json(TOPICS_FILE, posted_topics[-40:])
                                history.append(link)
                                new_posts_found = True
                            else:
                                print(f"⚠️ Groq не ответил — попробую эту статью на следующем прогоне.")
                        else:
                            history.append(link)
                            new_posts_found = True
                        time.sleep(2)
        except Exception as e:
            print(f"Ошибка при обработке ленты {rss_url}: {e}")

    if new_posts_found:
        save_json(HISTORY_FILE, history[-500:])
    
    if is_first_run:
        print("\n=== Инициализация завершена. Теперь жду новые посты. ===\n")
    elif not new_posts_found:
        print("Новых постов пока нет.")
    FIRST_RUN = False

if __name__ == '__main__':
    print("AI-Парсер запущен!")

    if os.environ.get('RUN_ONCE', '').lower() == 'true':
        check_news()
    else:
        while True:
            try:
                check_news()
                print(f"Сплю {CHECK_INTERVAL} секунд...\n")
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                print("Остановка скрипта...")
                break
            except Exception as e:
                print(f"Критическая ошибка: {e}")
                time.sleep(60)
