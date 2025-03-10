import os
import logging
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.utils import executor
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Получаем токен из переменной окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не найден в переменных окружения")

# URL для API сервиса с рецептами
FASTAPI_URL = "http://fastapi_app:8000/generate_recipe"

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


# Обработчик текстовых сообщений
@dp.message_handler(state=RecipeStates.waiting_for_query, content_types=types.ContentTypes.TEXT)
async def process_recipe_request(message: types.Message, state: FSMContext):
    user_query = message.text

    # Отправка "печатает" статуса
    await bot.send_chat_action(message.chat.id, 'typing')

    # Отправляем сообщение о начале обработки
    processing_msg = await message.reply("🔍 Ищу рецепт... Это может занять несколько секунд.")

    try:
        # Используем requests вместо aiohttp
        response = requests.post(
            FASTAPI_URL,
            json={"query": user_query, "max_length": 2048, "temperature": 0.7},
            timeout=60  # Устанавливаем timeout для запроса
        )

        if response.status_code == 200:
            result = response.json()
            recipe_text = result.get("recipe", "К сожалению, не удалось сгенерировать рецепт.")

            # Разбиваем длинный текст на части, если необходимо
            if len(recipe_text) > 4000:
                parts = [recipe_text[i:i + 4000] for i in range(0, len(recipe_text), 4000)]
                for part in parts:
                    await message.reply(part, parse_mode=ParseMode.MARKDOWN)
            else:
                await message.reply(recipe_text, parse_mode=ParseMode.MARKDOWN)
        else:
            error_text = response.text
            logging.error(f"Ошибка API: {response.status_code} - {error_text}")
            await message.reply(
                "😞 Произошла ошибка при получении рецепта. Попробуйте другой запрос или повторите позже.")
    except requests.exceptions.Timeout:
        logging.error("Timeout при запросе к API")
        await message.reply(
            "⏱️ Запрос занимает слишком много времени. Пожалуйста, попробуйте позже или уточните запрос.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при запросе к API: {e}")
        await message.reply("😞 Не удалось связаться с сервисом рецептов. Пожалуйста, попробуйте позже.")
    except Exception as e:
        logging.error(f"Непредвиденная ошибка: {e}")
        await message.reply("😞 Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.")
    finally:
        # Удаляем сообщение о обработке
        await bot.delete_message(chat_id=processing_msg.chat.id, message_id=processing_msg.message_id)


# Обработчик всех остальных сообщений
@dp.message_handler(content_types=types.ContentTypes.ANY, state='*')
async def unknown_message(message: types.Message):
    await message.reply(
        "Я понимаю только текстовые сообщения с запросами о рецептах. "
        "Пожалуйста, напишите, какое блюдо вы хотели бы приготовить."
    )


# Запуск бота
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)