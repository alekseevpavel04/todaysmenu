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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Get token from environment variable
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not found in environment variables")

# URL for the recipe service API
FASTAPI_URL = "http://fastapi_app:8000/generate_recipe"
HEALTH_CHECK_URL = "http://fastapi_app:8000/health"


# Function to check API availability
def is_api_ready():
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        data = response.json()
        if data["status"] == "ok":
            return True
        logger.info(f"API is not ready yet: {data}")
        return False
    except Exception as e:
        logger.warning(f"Error checking API readiness: {e}")
        return False


# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# Define states for FSM
class RecipeStates(StatesGroup):
    waiting_for_query = State()


# Handler for /start command
@dp.message_handler(commands=['start'], state='*')
async def send_welcome(message: types.Message):
    await message.reply(
        "👨‍🍳 Welcome to the culinary bot!\n\n"
        "I can help you cook different dishes. "
        "Just write what you would like to cook, "
        "and I will provide you with a detailed recipe.\n\n"
        "For example: 'How to cook carbonara?' or 'Tiramisu recipe'"
    )
    await RecipeStates.waiting_for_query.set()


# Handler for /help command
@dp.message_handler(commands=['help'], state='*')
async def send_help(message: types.Message):
    await message.reply(
        "🔍 How to use the bot:\n\n"
        "1. Simply write the dish name or a cooking query\n"
        "2. Wait for the response with the recipe\n"
        "3. Enjoy the cooking process!\n\n"
        "Example queries:\n"
        "• How to cook borsch?\n"
        "• Apple charlotte recipe\n"
        "• I want to cook pasta carbonara\n"
        "• Homemade bread recipe"
    )


# Handler for /status command - new command to check API status
@dp.message_handler(commands=['status'], state='*')
async def check_api_status(message: types.Message):
    await message.reply("🔄 Checking recipe service status...")
    if is_api_ready():
        await message.reply("✅ Recipe service is working normally and ready to use!")
    else:
        await message.reply("⚠️ Recipe service is currently unavailable or loading. Try again later.")


# Handler for text messages
@dp.message_handler(state=RecipeStates.waiting_for_query, content_types=types.ContentTypes.TEXT)
async def process_recipe_request(message: types.Message, state: FSMContext):
    user_query = message.text

    # Check API readiness before sending request
    if not is_api_ready():
        await message.reply(
            "⚠️ Recipe service is currently loading or unavailable. "
            "Please try again in a few minutes."
        )
        return

    # Send "typing" status
    await bot.send_chat_action(message.chat.id, 'typing')

    # Send message about processing start
    processing_msg = await message.reply("🔍 Searching for recipe... This may take a few seconds.")

    try:
        # Make a request to the API with increased timeout
        response = requests.post(
            FASTAPI_URL,
            json={"query": user_query, "max_length": 2048, "temperature": 0.7},
            timeout=120  # Increase timeout to 2 minutes for generation
        )

        if response.status_code == 200:
            result = response.json()
            recipe_text = result.get("recipe", "Unfortunately, I couldn't generate the recipe.")

            # Split long text into parts if necessary
            if len(recipe_text) > 4000:
                # Delete processing message
                await bot.delete_message(chat_id=processing_msg.chat.id, message_id=processing_msg.message_id)

                # Send message about long recipe
                await message.reply("📖 The recipe is quite long, sending in parts:")

                parts = [recipe_text[i:i + 4000] for i in range(0, len(recipe_text), 4000)]
                for i, part in enumerate(parts):
                    await message.reply(f"Part {i + 1}/{len(parts)}:\n\n{part}", parse_mode=ParseMode.MARKDOWN)
            else:
                # Delete processing message
                await bot.delete_message(chat_id=processing_msg.chat.id, message_id=processing_msg.message_id)
                await message.reply(recipe_text, parse_mode=ParseMode.MARKDOWN)
        elif response.status_code == 503:
            # Special handling for the case when the model is still loading
            logger.warning("API responded with 503 - service is still loading")
            await message.reply(
                "⏳ The language model is still loading. Please wait a few minutes and try again.")
        else:
            error_text = response.text
            logger.error(f"API Error: {response.status_code} - {error_text}")
            await message.reply(
                "😞 An error occurred while getting the recipe. Try another query or try again later.")
    except requests.exceptions.Timeout:
        logger.error("Timeout when requesting API")
        await message.reply(
            "⏱️ The request is taking too long. Please try again later or refine your query.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error when requesting API: {e}")
        await message.reply("😞 Could not connect to the recipe service. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await message.reply("😞 An unexpected error occurred. Please try again later.")
    finally:
        # Check if the processing message still exists and delete it if yes
        try:
            await bot.delete_message(chat_id=processing_msg.chat.id, message_id=processing_msg.message_id)
        except:
            pass  # If the message is already deleted, ignore the error


# Handler for all other messages
@dp.message_handler(content_types=types.ContentTypes.ANY, state='*')
async def unknown_message(message: types.Message):
    await message.reply(
        "I only understand text messages with recipe queries. "
        "Please write what dish you would like to cook."
    )


# Start the bot with waiting for API readiness
if __name__ == '__main__':
    logger.info("Starting Telegram bot...")

    # Wait until API is ready
    retry_count = 0
    max_retries = 30  # Maximum 5 minutes of waiting (30 * 10 seconds)
    retry_interval = 10  # 10 seconds between attempts

    logger.info("Checking API availability before starting the bot...")
    while not is_api_ready() and retry_count < max_retries:
        logger.info(f"Waiting for FastAPI service to be ready... Attempt {retry_count + 1}/{max_retries}")
        time.sleep(retry_interval)
        retry_count += 1

    if retry_count >= max_retries:
        logger.warning(
            "Could not wait for FastAPI service to be ready after several attempts. Starting the bot anyway.")
    else:
        logger.info("✅ FastAPI service is ready!")

    # Start the bot
    executor.start_polling(dp, skip_updates=True)