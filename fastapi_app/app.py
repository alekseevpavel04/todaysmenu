import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Отключаем предупреждения Python
os.environ["PYTHONWARNINGS"] = "ignore"

# Инициализация FastAPI приложения
app = FastAPI(title="Кулинарный помощник", description="API для генерации рецептов")


# Определяем модель запроса
class RecipeRequest(BaseModel):
    query: str
    max_length: Optional[int] = 1024
    temperature: Optional[float] = 0.7


# Определяем модель ответа
class RecipeResponse(BaseModel):
    recipe: str


# Загружаем модель и токенизатор
@app.on_event("startup")
async def startup_event():
    global model, tokenizer

    model_name = "Qwen/Qwen2.5-0.5B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print(f"Модель {model_name} успешно загружена")
        print(f"Используемое устройство: {model.device}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        raise


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
    try:
        prompt = create_chef_prompt(request.query)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=request.max_length,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        recipe_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Убираем промпт из ответа
        recipe_text = recipe_text.replace(prompt, "").strip()

        return RecipeResponse(recipe=recipe_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации рецепта: {str(e)}")


# Проверка состояния сервиса
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)