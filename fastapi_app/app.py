import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import asynccontextmanager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Отключаем предупреждения Python
os.environ["PYTHONWARNINGS"] = "ignore"

# Глобальные переменные для модели и токенизатора
model = None
tokenizer = None


# Создаем контекстный менеджер для обработки событий жизненного цикла приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Declare global variables first
    global model, tokenizer

    logger.info("==== Начало загрузки модели ====")
    model_name = "Qwen/Qwen2.5-0.5B"

    try:
        logger.info(f"Загрузка токенизатора {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Токенизатор успешно загружен")

        logger.info(f"Загрузка модели {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        logger.info(f"✓ Модель {model_name} успешно загружена")
        logger.info(f"✓ Используемое устройство: {model.device}")
    except Exception as e:
        logger.error(f"✗ Критическая ошибка при загрузке модели: {str(e)}")
        raise

    logger.info("==== Загрузка модели завершена ====")

    yield  # Здесь приложение работает

    # Код, выполняемый при завершении приложения
    logger.info("Завершение работы приложения, освобождение ресурсов...")
    # Здесь не нужно повторно объявлять global, так как вы уже это сделали выше
    del model
    del tokenizer
    torch.cuda.empty_cache()


# Инициализация FastAPI приложения с менеджером жизненного цикла
app = FastAPI(
    title="Кулинарный помощник",
    description="API для генерации рецептов",
    lifespan=lifespan
)


# Определяем модель запроса
class RecipeRequest(BaseModel):
    query: str
    max_length: Optional[int] = 1024
    temperature: Optional[float] = 0.7


# Определяем модель ответа
class RecipeResponse(BaseModel):
    recipe: str


# Промпт для роли повара
def create_chef_prompt(query):
    return f"""Ты - опытный шеф-повар с многолетним стажем. Твоя задача - помогать людям готовить вкусные блюда, предоставляя подробные рецепты.

Пользователь хочет приготовить: {query}

Пожалуйста, предоставь рецепт, который включает:
1. Список ингредиентов с точными пропорциями
2. Подробные шаги приготовления
3. Примерное время приготовления
4. Советы по подаче и хранению блюда

Рецепт:
"""


# Эндпоинт для генерации рецепта
@app.post("/generate_recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Модель еще не загружена. Пожалуйста, попробуйте позже."
        )

    try:
        logger.info(f"Получен запрос на генерацию рецепта: {request.query}")
        prompt = create_chef_prompt(request.query)

        # When tokenizing input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to(model.device)
        logger.info(f"Начинаю генерацию текста...")

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        recipe_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Убираем промпт из ответа
        recipe_text = recipe_text.replace(prompt, "").strip()

        logger.info(f"Рецепт успешно сгенерирован (длина: {len(recipe_text)} символов)")
        return RecipeResponse(recipe=recipe_text)
    except Exception as e:
        logger.error(f"Ошибка при генерации рецепта: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации рецепта: {str(e)}")


# Проверка состояния сервиса
@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        logger.warning("Запрос к /health, но модель еще не загружена!")
        return {"status": "loading", "message": "Model is still loading"}
    return {"status": "ok", "message": "Service is fully operational"}


# Запуск сервера
if __name__ == "__main__":
    logger.info("Запуск сервера FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=8000)