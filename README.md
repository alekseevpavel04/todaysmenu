# Today's menu

The "Today's menu" project is a Telegram bot that leverages Language Models (LLMs) to assist users in selecting recipes from [allrecipes.com](https://www.allrecipes.com) and provides personalized recommendations from the Chef.

## Example
![example](https://github.com/user-attachments/assets/77ede96d-399b-49f6-93da-5c5b5f566e46)


## Project Structure
```
/
├── docker-compose.yml
├── .env
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── fastapi_app.py
├── telegram_bot/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── bot.py
└── benchmark/
    ├── Dockerfile
    ├── requirements.txt
    └── benchmark.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alekseevpavel04/what-to-cook.git
   cd what-to-cook
   ```

2. Create a `.env` file with your Telegram token:
   ```
   TELEGRAM_TOKEN=YOUR_TOKEN
   ```

3. Build and start the project:
   ```
   docker-compose up --build telegram_bot fastapi_app
   ```

4. To run benchmarks:
   ```
   docker-compose up --build benchmark
   ```


## Usage

Once the bot is running, users can interact with it on Telegram to get recipe recommendations. The bot uses the FastAPI backend and LLM to provide personalized suggestions based on user preferences.
