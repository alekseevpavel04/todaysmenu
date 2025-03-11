import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import asynccontextmanager
import requests
from bs4 import BeautifulSoup
import re
import json
from fastapi.responses import StreamingResponse
import asyncio

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
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

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
    description="API for recipe generation and comparison",
    lifespan=lifespan
)


# Define the request model
class RecipeRequest(BaseModel):
    query: str
    max_length: Optional[int] = 1024
    temperature: Optional[float] = 0.7


# Headers for web requests to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}


# Function to use LLM to optimize search query
async def optimize_search_query(query: str) -> str:
    global model, tokenizer

    if model is None or tokenizer is None:
        logger.warning("Model not loaded for query optimization, using fallback cleaning")
        # Fallback to simple cleaning if model is not available
        return clean_search_query(query)

    logger.info(f"Using LLM to optimize query: {query}")

    # Create system message with clearer instructions and more emphasis
    system_message = {
        "role": "system",
        "content": (
            "You are a culinary search assistant. Your task is to extract the main dish name from user cooking queries and convert it into an optimal search term. "
            "Follow these rules STRICTLY:\n"
            "1. Remove unnecessary words like 'how to cook', 'recipe for', 'best way to prepare', 'homemade', etc.\n"
            "2. Fix obvious spelling mistakes (e.g., 'spagheti' → 'spaghetti').\n"
            "3. Extract ONLY the dish name (e.g., 'Recipe for delicious chicken parmesan' → 'chicken parmesan').\n"
            "4. Do not add any extra words that were not in the original query.\n"
            "5. No punctuation, capitalization, or extra formatting in your response.\n"
            "6. Your response must be extremely short: just 1-3 words.\n\n"
            "Examples:\n"
            "- 'How to cook lasagna at home?' → 'lasagna'\n"
            "- 'Best way to prepare beef stroganoff' → 'beef stroganoff'\n"
            "- 'Recipe for easy chicken tikka masala' → 'chicken tikka masala'\n"
            "- 'Home made spagheti carbonara' → 'spaghetti carbonara'\n"
            "- 'How do I make a classic apple pie?' → 'apple pie'\n"
        )
    }

    # Create user message with query
    user_message = {
        "role": "user",
        "content": f"Convert this cooking query to optimal search terms: {query}"
    }

    # Apply chat template
    messages = [system_message, user_message]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the text
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response with stricter parameters for more deterministic output
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=15,  # Shorter response limit
        temperature=0.1,  # Much lower temperature for more deterministic response
        do_sample=False,  # Turn off sampling for more deterministic output
        top_p=0.95,
        repetition_penalty=1.2  # Discourage repetition
    )

    # Extract only the generated tokens (without input tokens)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the response
    optimized_query = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Clean the response further (remove quotes, trim, etc.)
    optimized_query = optimized_query.strip().strip('"\'').strip()

    # Apply additional post-processing to catch common issues
    # Remove any added words like "preparation method", "recipe", etc.
    common_additions = ["preparation", "method", "recipe", "cooking", "instructions", "how to", "how to make"]
    for addition in common_additions:
        optimized_query = re.sub(r'\s+' + addition + r'\s*$', '', optimized_query, flags=re.IGNORECASE)
        optimized_query = re.sub(r'^' + addition + r'\s+', '', optimized_query, flags=re.IGNORECASE)

    # Ensure the query is clean and simple
    optimized_query = re.sub(r'\s+', ' ', optimized_query).strip()

    # Final verification - if the optimized query is longer than the original,
    # it might have added unwanted terms, so use the fallback instead
    if len(optimized_query.split()) > len(query.split()) + 1:
        logger.warning(f"Optimized query too verbose, using fallback: '{optimized_query}'")
        optimized_query = clean_search_query(query)

    logger.info(f"LLM optimized query: '{query}' -> '{optimized_query}'")
    return optimized_query


# Function to clean and optimize search query (fallback if LLM not available)
def clean_search_query(query: str) -> str:
    # Remove "how to cook", "recipe for", etc.
    cleaned_query = re.sub(r'how to (make|cook|prepare)|recipe for|how do i make', '', query, flags=re.IGNORECASE)
    # Remove question marks
    cleaned_query = cleaned_query.replace('?', '')
    # Remove extra spaces and strip
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    logger.info(f"Fallback cleaned query: '{query}' -> '{cleaned_query}'")
    return cleaned_query


# Function to search AllRecipes and get multiple recipes
async def search_multiple_recipes(query: str, max_recipes: int = 3) -> List[Dict]:
    try:
        # Use LLM to optimize the search query
        cleaned_query = await optimize_search_query(query)

        logger.info(f"Searching AllRecipes for: {cleaned_query}, max recipes: {max_recipes}")

        # Step 1: Search for recipes
        search_url = f"https://www.allrecipes.com/search?q={cleaned_query.replace(' ', '+')}"
        search_response = requests.get(search_url, headers=HEADERS, timeout=15)
        search_response.raise_for_status()

        search_soup = BeautifulSoup(search_response.text, 'html.parser')

        # Find recipe links
        recipe_links = []
        recipe_titles = []

        # Look for cards with recipes
        for card in search_soup.find_all(['a'], href=True):
            href = card['href']

            # Check if it's a recipe link
            if '/recipe/' in href and not href.endswith('/reviews/') and not href.endswith('/photos/'):
                # Only add new links (avoid duplicates)
                if href not in recipe_links:
                    # Try to find the title
                    title = None

                    # Try to extract title from the card
                    title_element = card.find('h3', class_='card__title')
                    if title_element:
                        title = title_element.text.strip()
                    else:
                        # Try other potential ways to get the title
                        title_element = card.find(['h1', 'h2', 'h3', 'h4'])
                        if title_element:
                            title = title_element.text.strip()

                    # If we didn't find a title in the card, we'll fetch it directly
                    recipe_links.append(href)
                    if title:
                        recipe_titles.append(title)
                    else:
                        recipe_titles.append(None)  # We'll fetch this later

                    if len(recipe_links) >= max_recipes:
                        break

        if not recipe_links:
            logger.info(f"No recipes found for query: {cleaned_query}")
            return []

        logger.info(f"Found {len(recipe_links)} recipe links")

        # Create recipes with titles - fetch missing titles if needed
        recipes = []
        for i, (link, title) in enumerate(zip(recipe_links, recipe_titles)):
            if title is None:
                # We need to fetch the title directly from the recipe page
                recipe_details = await extract_recipe_details(link)
                title = recipe_details["title"]

            recipes.append({
                "title": title,
                "link": link,
                "index": i + 1
            })

        return recipes

    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        return []


# Function to extract detailed recipe information
async def extract_recipe_details(recipe_link: str) -> Dict:
    try:
        logger.info(f"Fetching recipe from: {recipe_link}")
        recipe_response = requests.get(recipe_link, headers=HEADERS, timeout=10)
        recipe_response.raise_for_status()

        recipe_soup = BeautifulSoup(recipe_response.text, 'html.parser')

        # Extract recipe title
        title_element = recipe_soup.find('h1', class_='article-heading')
        title = title_element.text.strip() if title_element else "Unknown Recipe"

        # Extract ingredients
        ingredients = []
        ingredients_section = recipe_soup.find('div', {'id': 'mntl-structured-ingredients_1-0'})
        if ingredients_section:
            for item in ingredients_section.find_all('li', class_='mntl-structured-ingredients__list-item'):
                ingredients.append(item.text.strip())

        # Extract preparation steps
        preparation = []
        steps_section = recipe_soup.find('div', {'id': re.compile(r'mntl-sc-block_\d+-0')})
        if steps_section:
            step_idx = 1
            for step in steps_section.find_all('p', class_='comp mntl-sc-block'):
                step_text = step.text.strip()
                if step_text:  # Only add non-empty steps
                    preparation.append(f"{step_idx}. {step_text}")
                    step_idx += 1

        # Extract timing information
        cooking_time = "Not specified"
        timing_section = recipe_soup.find('div', class_='mntl-recipe-details__content')
        if timing_section:
            time_elements = timing_section.find_all('div', class_='mntl-recipe-details__item')
            for element in time_elements:
                label = element.find('div', class_='mntl-recipe-details__label')
                value = element.find('div', class_='mntl-recipe-details__value')
                if label and value and (
                        'time' in label.text.lower() or 'total' in label.text.lower() or 'cook' in label.text.lower()):
                    cooking_time = f"{label.text.strip()}: {value.text.strip()}"
                    break  # Use the first time-related information

        return {
            "title": title,
            "ingredients": ingredients,
            "preparation_steps": preparation,
            "cooking_time": cooking_time,
            "source_url": recipe_link
        }

    except Exception as e:
        logger.error(f"Error fetching recipe details from {recipe_link}: {str(e)}")
        return {
            "title": "Error fetching recipe",
            "ingredients": [],
            "preparation_steps": [],
            "cooking_time": "Unknown",
            "source_url": recipe_link
        }


# Function to generate recipe analysis using the model
async def generate_recipe_analysis(recipe: Dict, query: str, max_length: int = 256,
                                   temperature: float = 0.7) -> str:
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not yet loaded. Please try again later."
        )

    logger.info(f"Generating recipe analysis for: {recipe['title']}")

    # Create system message with emphasis on brevity
    system_message = {
        "role": "system",
        "content": (
            "You are a professional chef providing very concise recipe insights. Examine the entire recipe "
            "carefully and provide a brief comment (just 1-2 sentences) that highlights one or two notable aspects "
            "of this recipe. This could be about unique ingredients, cooking techniques, cultural origins, or how "
            "it differs from traditional versions. BE EXTREMELY BRIEF - your entire response must not exceed 2 sentences."
        )
    }

    # Include complete recipe details for better analysis
    ingredients_text = "\n".join([f"• {ing}" for ing in recipe['ingredients']])
    steps_text = "\n".join(recipe['preparation_steps'])

    # Create user message with the complete recipe details
    user_message = {
        "role": "user",
        "content": (
            f"Here's a recipe for {query} that I'd like you to briefly comment on:\n\n"
            f"Title: {recipe['title']}\n\n"
            f"Cooking time: {recipe['cooking_time']}\n\n"
            f"Ingredients:\n{ingredients_text}\n\n"
            f"Preparation Steps:\n{steps_text}\n\n"
            f"Please provide 1-2 sentences about what makes this recipe interesting or unique."
        )
    }

    # Apply chat template
    messages = [system_message, user_message]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    logger.info("Chat template for analysis successfully applied")

    # Tokenize the text
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    logger.info(f"Starting recipe analysis text generation...")

    # Generate response with stricter parameters for shorter output
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2
    )

    # Extract only the generated tokens (without input tokens)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the response
    analysis_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Ensure the response is brief - truncate if needed
    sentences = re.split(r'(?<=[.!?])\s+', analysis_text.strip())
    if len(sentences) > 2:
        analysis_text = ' '.join(sentences[:2])

    logger.info(f"Recipe analysis successfully generated (length: {len(analysis_text)} characters)")
    return analysis_text


# Streaming response generator for multiple recipes
async def recipe_stream_generator(recipe_request: RecipeRequest):
    try:
        query = recipe_request.query
        logger.info(f"Streaming recipes for query: {query}")

        # Find recipes - just get basic info (title and link)
        # Explicitly set max_recipes to 3 to ensure we don't fetch more than needed
        recipes = await search_multiple_recipes(query, max_recipes=3)

        if not recipes:
            yield json.dumps({"error": f"No recipes found for '{query}'"}) + "\n"
            return

        # For each recipe, fetch details and generate analysis
        for recipe in recipes:
            # Get detailed recipe info
            detailed_recipe = await extract_recipe_details(recipe['link'])

            # Generate analysis for this recipe
            recipe_analysis = await generate_recipe_analysis(
                detailed_recipe,
                query,
                max_length=256,  # Shorter length for brief comments
                temperature=recipe_request.temperature
            )

            # Yield the recipe info and analysis
            yield json.dumps({
                "type": "recipe",
                "content": f"# Recipe {recipe['index']}: {detailed_recipe['title']}\n\nLink: {detailed_recipe['source_url']}\n\n**Chef's Note:** {recipe_analysis}"
            }) + "\n"

            # Small delay between messages to ensure proper ordering
            await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in streaming recipes: {str(e)}")
        yield json.dumps({"error": f"Error processing request: {str(e)}"}) + "\n"


# Endpoint for streaming recipes and comparison
@app.post("/stream_recipes")
async def stream_recipes(request: RecipeRequest):
    return StreamingResponse(
        recipe_stream_generator(request),
        media_type="text/event-stream"
    )


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