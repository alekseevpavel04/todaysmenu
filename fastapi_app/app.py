import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Disable Python warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# Global variables for model and tokenizer
model = None
tokenizer = None

logger.info(f"CUDA is available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")


# Create a context manager to handle application lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Declare global variables first
    global model, tokenizer

    logger.info("==== Starting model loading ====")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    try:
        logger.info(f"Loading tokenizer {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer successfully loaded")

        logger.info(f"Loading model {model_name}...")

        # Load the model with correct parameters
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",  # automatically selects the appropriate type
            device_map="auto"  # automatically places on available devices
        )
        logger.info(f"✓ Model {model_name} successfully loaded")
        logger.info(f"✓ Device used: {model.device}")
    except Exception as e:
        logger.error(f"✗ Critical error during model loading: {str(e)}")
        raise

    logger.info("==== Model loading completed ====")

    yield  # Here the application runs

    # Code executed when the application is shutting down
    logger.info("Terminating the application, freeing resources...")
    # No need to redeclare global here, as you already did above
    del model
    del tokenizer
    torch.cuda.empty_cache()


# Initialize FastAPI application with lifecycle manager
app = FastAPI(
    title="Culinary Assistant",
    description="API for recipe generation",
    lifespan=lifespan
)


# Define the request model
class RecipeRequest(BaseModel):
    query: str
    max_length: Optional[int] = 1024
    temperature: Optional[float] = 0.7


# Define the response model
class RecipeResponse(BaseModel):
    recipe: str


# Function to create messages in the format expected by Qwen
def create_chef_messages(query: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a professional chef who specializes in culinary arts and recipes. "
                       "You provide detailed recipes with precise ingredient proportions and preparation steps."
                       "You speak only English."
        },
        {
            "role": "user",
            "content": f"I want to cook {query}. Please give me a detailed recipe."
        },
        {
            "role": "assistant",
            "content": f"I'm happy to help you with a recipe for {query}!"
        },
        {
            "role": "user",
            "content": f"Write me a recipe for {query} in the following format:\n\n"
                       f"# Recipe: {query}\n\n"
                       f"## Ingredients:\n"
                       f"[list of ingredients with exact proportions]\n\n"
                       f"## Preparation:\n"
                       f"[numbered preparation steps]\n\n"
                       f"## Cooking time:\n"
                       f"## Serving and storage tips:\n"
                       f"[useful tips]"
        }
    ]


# Endpoint for recipe generation
@app.post("/generate_recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not yet loaded. Please try again later."
        )

    try:
        logger.info(f"Received request for recipe generation: {request.query}")

        # Create messages in the format expected by Qwen
        messages = create_chef_messages(request.query)

        # Apply chat_template to messages
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        logger.info("Chat template successfully applied")

        # Tokenize the text
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        logger.info(f"Starting text generation...")

        # Generate response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.9
        )

        # Extract only the generated tokens (without input tokens)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the response
        recipe_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info(f"Recipe successfully generated (length: {len(recipe_text)} characters)")
        return RecipeResponse(recipe=recipe_text)

    except Exception as e:
        logger.error(f"Error during recipe generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during recipe generation: {str(e)}")


# Service health check
@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        logger.warning("Request to /health, but the model is not yet loaded!")
        return {"status": "loading", "message": "Model is still loading"}
    return {"status": "ok", "message": "Service is fully operational"}


# Start the server
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)