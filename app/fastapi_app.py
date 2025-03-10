# Сервис должен быть LLM моделью - deepseek самая легкая (языковая)
# У нее должна быть роль - Повар
# Она будет принимать запросы пользователя что он хочет приготовить и выдавать рецепты


os.environ["PYTHONWARNINGS"] = "ignore"

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn fastapi_app:app --reload