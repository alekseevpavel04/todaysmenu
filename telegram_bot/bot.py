import os
import logging
import requests
import time
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.utils import executor
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Получаем токен из переменной окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не найден в переменных окружения")

# URL для API сервиса с рецептами
FASTAPI_URL = "http://fastapi_app:8000/generate_recipe"
HEALTH_CHECK_URL = "http://fastapi_app:8000/health"


# Функция для проверки доступности API
def is_api_ready():
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        data = response.json()
        if data["status"] == "ok":
            return True
        logger.info(f"API еще не готов: {data}")
        return False
    except Exception as e:
        logger.warning(f"Ошибка при проверке готовности API: {e}")
        return False


# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# Определение состояний для FSM
class RecipeStates(StatesGroup):
    waiting_for_query = State()


# Обработчик команды /start
@dp.message_handler(commands=['start'], state='*')
async def send_welcome(message: types.Message):
    await message.reply(
        "👨‍🍳 Приветствую вас в кулинарном боте!\n\n"
        "Я могу помочь вам приготовить разные блюда. "
        "Просто напишите, что вы хотели бы приготовить, "
        "и я предоставлю вам подробный рецепт.\n\n"
        "Например: 'Как приготовить карбонару?' или 'Рецепт тирамису'"
    )
    await RecipeStates.waiting_for_query.set()


# Обработчик команды /help
@dp.message_handler(commands=['help'], state='*')
async def send_help(message: types.Message):
    await message.reply(
        "🔍 Как пользоваться ботом:\n\n"
        "1. Просто напишите название блюда или запрос о приготовлении\n"
        "2. Дождитесь ответа с рецептом\n"
        "3. Наслаждайтесь процессом приготовления!\n\n"
        "Примеры запросов:\n"
        "• Как приготовить борщ?\n"
        "• Рецепт шарлотки с яблоками\n"
        "• Хочу приготовить пасту карбонара\n"
        "• Рецепт хлеба в домашних условиях"
    )


# Обработчик команды /status - новая команда для проверки статуса API
@dp.message_handler(commands=['status'], state='*')
async def check_api_status(message: types.Message):
    await message.reply("🔄 Проверяю состояние сервиса рецептов...")
    if is_api_ready():
        await message.reply("✅ Сервис рецептов работает нормально и готов к использованию!")
    else:
        await message.reply("⚠️ Сервис рецептов сейчас недоступен или загружается. Попробуйте позже.")


# Обработчик текстовых сообщений
@dp.message_handler(state=RecipeStates.waiting_for_query, content_types=types.ContentTypes.TEXT)
async def process_recipe_request(message: types.Message, state: FSMContext):
    user_query = message.text

    # Проверяем готовность API перед отправкой запроса
    if not is_api_ready():
        await message.reply(
            "⚠️ Сервис рецептов сейчас загружается или недоступен. "
            "Пожалуйста, попробуйте через несколько минут."
        )
        return

    # Отправка "печатает" статуса
    await bot.send_chat_action(message.chat.id, 'typing')

    # Отправляем сообщение о начале обработки
    processing_msg = await message.reply("🔍 Ищу рецепт... Это может занять несколько секунд.")

    try:
        # Делаем запрос к API с увеличенным таймаутом
        response = requests.post(
            FASTAPI_URL,
            json={"query": user_query, "max_length": 2048, "temperature": 0.7},
            timeout=120  # Увеличиваем таймаут до 2 минут для генерации
        )

        if response.status_code == 200:
            result = response.json()
            recipe_text = result.get("recipe", "К сожалению, не удалось сгенерировать рецепт.")

            # Разбиваем длинный текст на части, если необходимо
            if len(recipe_text) > 4000:
                # Удаляем сообщение о обработке
                await bot.delete_message(chat_id=processing_msg.chat.id, message_id=processing_msg.message_id)

                # Отправляем сообщение о длинном рецепте
                await message.reply("📖 Рецепт получился большим, отправляю по частям:")

                parts = [recipe_text[i:i + 4000] for i in range(0, len(recipe_text), 4000)]
                for i, part in enumerate(parts):
                    await message.reply(f"Часть {i + 1}/{len(parts)}:\n\n{part}", parse_mode=ParseMode.MARKDOWN)
            else:
                # Удаляем сообщение о обработке
                await bot.delete_message(chat_id=processing_msg.chat.id, message_id=processing_msg.message_id)
                await message.reply(recipe_text, parse_mode=ParseMode.MARKDOWN)
        elif response.status_code == 503:
            # Специальная обработка для случая, когда модель еще загружается
            logger.warning("API ответило 503 - сервис еще загружается")
            await message.reply(
                "⏳ Языковая модель еще загружается. Пожалуйста, подождите несколько минут и попробуйте снова.")
        else:
            error_text = response.text
            logger.error(f"Ошибка API: {response.status_code} - {error_text}")
            await message.reply(
                "😞 Произошла ошибка при получении рецепта. Попробуйте другой запрос или повторите позже.")
    except requests.exceptions.Timeout:
        logger.error("Timeout при запросе к API")
        await message.reply(
            "⏱️ Запрос занимает слишком много времени. Пожалуйста, попробуйте позже или уточните запрос.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе к API: {e}")
        await message.reply("😞 Не удалось связаться с сервисом рецептов. Пожалуйста, попробуйте позже.")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        await message.reply("😞 Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.")
    finally:
        # Проверяем, существует ли еще сообщение о обработке, и удаляем его если да
        try:
            await bot.delete_message(chat_id=processing_msg.chat.id, message_id=processing_msg.message_id)
        except:
            pass  # Если сообщение уже удалено, игнорируем ошибку


# Обработчик всех остальных сообщений
@dp.message_handler(content_types=types.ContentTypes.ANY, state='*')
async def unknown_message(message: types.Message):
    await message.reply(
        "Я понимаю только текстовые сообщения с запросами о рецептах. "
        "Пожалуйста, напишите, какое блюдо вы хотели бы приготовить."
    )


# Запуск бота с ожиданием готовности API
if __name__ == '__main__':
    logger.info("Запуск Telegram бота...")

    # Ждем, пока API будет готов
    retry_count = 0
    max_retries = 30  # Максимум 5 минут ожидания (30 * 10 секунд)
    retry_interval = 10  # 10 секунд между попытками

    logger.info("Проверка доступности API перед запуском бота...")
    while not is_api_ready() and retry_count < max_retries:
        logger.info(f"Ожидание готовности FastAPI сервиса... Попытка {retry_count + 1}/{max_retries}")
        time.sleep(retry_interval)
        retry_count += 1

    if retry_count >= max_retries:
        logger.warning(
            "Не удалось дождаться готовности FastAPI сервиса после нескольких попыток. Запускаем бота в любом случае.")
    else:
        logger.info("✅ FastAPI сервис готов!")

    # Запускаем бота
    executor.start_polling(dp, skip_updates=True)