import asyncio
import logging
import time
import json
import re
import requests
import statistics
import sys
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
FASTAPI_URL = "http://fastapi_app:8000/stream_recipes"
HEALTH_CHECK_URL = "http://fastapi_app:8000/health"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Test queries
TEST_QUERIES = [
    "How to make chicken tikka masala",
    "Chocolate chip cookies recipe",
    "Best way to cook beef stroganoff",
    "Homemade pizza dough recipe",
    "Vegetarian lasagna recipe",
    "Classic tiramisu dessert",
    "Easy sushi rolls for beginners",
    "Traditional apple pie recipe",
    "Spicy pad thai noodles",
    "Creamy mushroom risotto"
]

# Benchmark results storage
benchmark_results = {
    "overall": {
        "total_time": 0,
        "success_rate": 0,
        "avg_response_time": 0,
        "avg_quality_score": 0,
    },
    "queries": []
}


async def is_api_ready() -> bool:
    """Check if the FastAPI service is ready"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(HEALTH_CHECK_URL, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "ok"
        return False
    except Exception as e:
        logger.warning(f"Error checking API readiness: {e}")
        return False


async def extract_recipe_content(url: str) -> str:
    """Extract text content from recipe URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=HEADERS, timeout=15) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Remove scripts and styles
                    for script in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                        script.extract()

                    text = soup.get_text(separator=' ', strip=True)
                    text = re.sub(r'\s+', ' ', text).strip()

                    # Truncate if too long
                    if len(text) > 15000:
                        text = text[:15000] + "..."

                    return text
                else:
                    logger.error(f"Failed to fetch recipe content. Status: {response.status}")
                    return ""
    except Exception as e:
        logger.error(f"Error extracting recipe content from {url}: {str(e)}")
        return ""


async def evaluate_chefs_note(recipe_text: str, chefs_note: str, model, tokenizer) -> Dict[str, Any]:
    """Use the LLM to evaluate the quality of the Chef's Note"""
    system_message = {
        "role": "system",
        "content": (
            "You are a culinary quality assessment expert. Your task is to evaluate the quality of a Chef's Note "
            "for a recipe. The Chef's Note should be insightful, accurate based on the recipe content, and provide "
            "valuable information about the recipe. Rate it on a scale from 1 to 10, where 1 is poor and 10 is excellent.\n\n"
            "Criteria for evaluation:\n"
            "1. Accuracy: Does the Chef's Note accurately reflect information in the recipe?\n"
            "2. Insight: Does it provide meaningful culinary insight or perspective?\n"
            "3. Uniqueness: Does it highlight something truly distinctive about this recipe?\n"
            "4. Brevity: Is it concise and to the point?\n\n"
            "Your response must be in JSON format with the following structure exactly:\n"
            "{\n"
            "  \"score\": <number between 1 and 10>,\n"
            "  \"strengths\": [\"<strength1>\", \"<strength2>\", ...],\n"
            "  \"weaknesses\": [\"<weakness1>\", \"<weakness2>\", ...],\n"
            "  \"explanation\": \"<brief explanation of the score>\"\n"
            "}\n\n"
            "The explanation should be BRIEF in just 1-2 sentences. Include at most 3 strengths and 3 weaknesses."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            f"Recipe Content:\n{recipe_text}\n\n"
            f"Chef's Note to evaluate:\n{chefs_note}\n\n"
            f"Provide your quality assessment in the required JSON format."
        )
    }

    messages = [system_message, user_message]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.2
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    evaluation_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Extract JSON from response
    try:
        # Find JSON pattern
        json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', evaluation_text)
        if json_match:
            json_str = json_match.group(0)
            evaluation = json.loads(json_str)
            # Ensure all required fields are present
            required_fields = ["score", "strengths", "weaknesses", "explanation"]
            for field in required_fields:
                if field not in evaluation:
                    evaluation[field] = "Missing" if field != "score" else 5
            # Ensure score is within range
            if not isinstance(evaluation["score"], (int, float)) or evaluation["score"] < 1 or evaluation["score"] > 10:
                evaluation["score"] = 5
            return evaluation
        else:
            logger.warning(f"Could not extract JSON from evaluation response")
            return {
                "score": 5,
                "strengths": ["Unable to extract proper evaluation"],
                "weaknesses": ["Unable to extract proper evaluation"],
                "explanation": "Failed to parse evaluation response"
            }
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error in evaluation: {str(e)}")
        return {
            "score": 5,
            "strengths": ["Unable to extract proper evaluation"],
            "weaknesses": ["Unable to extract proper evaluation"],
            "explanation": "Failed to parse evaluation JSON"
        }


async def process_query(query: str, model, tokenizer) -> Dict[str, Any]:
    """Process a single query and collect benchmark data"""
    query_results = {
        "query": query,
        "start_time": time.time(),
        "recipes": [],
        "errors": [],
        "total_response_time": 0,
        "successful_responses": 0
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    FASTAPI_URL,
                    json={"query": query, "max_length": 2048, "temperature": 0.7},
                    timeout=180
            ) as response:
                if response.status == 200:
                    # Process streaming response
                    buffer = b""
                    async for data, _ in response.content.iter_chunks():
                        buffer += data
                        if b"\n" in buffer:
                            lines = buffer.split(b"\n")
                            buffer = lines.pop()

                            for line in lines:
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    if "error" in data:
                                        query_results["errors"].append(data["error"])
                                    elif data.get("type") == "recipe":
                                        # Process recipe content
                                        content = data["content"]
                                        link_match = re.search(r'Link: (https?://[^\s]+)', content)
                                        chefs_note_match = re.search(r'\*\*Chef\'s Note:\*\* (.*?)(?:\n|$)', content)
                                        title_match = re.search(r'# Recipe \d+: (.*?)(?:\n|$)', content)

                                        recipe_data = {
                                            "content": content,
                                            "title": title_match.group(1) if title_match else "Unknown",
                                            "link": link_match.group(1) if link_match else None,
                                            "chefs_note": chefs_note_match.group(1) if chefs_note_match else None,
                                            "response_time": time.time() - query_results["start_time"],
                                            "evaluation": None
                                        }

                                        # If we have both link and chef's note, retrieve recipe content and evaluate
                                        if recipe_data["link"] and recipe_data["chefs_note"]:
                                            recipe_text = await extract_recipe_content(recipe_data["link"])
                                            if recipe_text:
                                                evaluation = await evaluate_chefs_note(
                                                    recipe_text,
                                                    recipe_data["chefs_note"],
                                                    model,
                                                    tokenizer
                                                )
                                                recipe_data["evaluation"] = evaluation
                                                recipe_data["quality_score"] = evaluation["score"]
                                            else:
                                                recipe_data["evaluation"] = {
                                                    "score": 0,
                                                    "strengths": [],
                                                    "weaknesses": ["Could not extract recipe content for evaluation"],
                                                    "explanation": "Recipe content extraction failed"
                                                }
                                                recipe_data["quality_score"] = 0

                                        query_results["recipes"].append(recipe_data)
                                        query_results["successful_responses"] += 1

                                except json.JSONDecodeError as e:
                                    logger.error(f"JSON decode error in streaming response: {e}")
                                    continue
                                except Exception as e:
                                    logger.error(f"Error processing streaming response line: {e}")
                                    continue
                else:
                    error_text = await response.text()
                    query_results["errors"].append(f"API Error: {response.status} - {error_text}")
    except asyncio.TimeoutError:
        query_results["errors"].append("Request timed out")
    except Exception as e:
        query_results["errors"].append(f"Error processing query: {str(e)}")

    query_results["end_time"] = time.time()
    query_results["total_time"] = query_results["end_time"] - query_results["start_time"]

    # Calculate quality metrics
    if query_results["recipes"]:
        quality_scores = [r.get("quality_score", 0) for r in query_results["recipes"] if "quality_score" in r]
        query_results["avg_quality_score"] = statistics.mean(quality_scores) if quality_scores else 0
    else:
        query_results["avg_quality_score"] = 0

    return query_results


async def run_benchmark():
    """Run the complete benchmark suite"""
    logger.info("Starting benchmark process")

    # Wait for API to be ready
    logger.info("Checking if FastAPI service is ready...")
    ready = await is_api_ready()
    if not ready:
        logger.error("FastAPI service is not ready. Aborting benchmark.")
        return

    logger.info("FastAPI service is ready. Loading model for evaluation...")

    # Load model for evaluation
    try:
        cache_path = "/root/.cache/huggingface"  # Using the same path as in the service
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        logger.info(f"Loading tokenizer {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_path
        )

        logger.info(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_path
        )
        logger.info(f"Model loaded successfully. Using device: {model.device}")
    except Exception as e:
        logger.error(f"Failed to load model for evaluation: {str(e)}")
        return

    # Run benchmark queries
    start_time = time.time()
    all_results = []

    logger.info(f"Starting to process {len(TEST_QUERIES)} test queries")
    for i, query in enumerate(TEST_QUERIES):
        logger.info(f"Processing query {i + 1}/{len(TEST_QUERIES)}: '{query}'")
        result = await process_query(query, model, tokenizer)
        all_results.append(result)
        logger.info(f"Query {i + 1} completed in {result['total_time']:.2f}s with {len(result['recipes'])} recipes")

        # Add a small delay between queries to avoid overwhelming the service
        if i < len(TEST_QUERIES) - 1:
            await asyncio.sleep(2)

    # Calculate overall metrics
    total_time = time.time() - start_time
    successful_queries = sum(1 for r in all_results if r["successful_responses"] > 0)
    success_rate = successful_queries / len(TEST_QUERIES) if TEST_QUERIES else 0

    response_times = []
    quality_scores = []

    for result in all_results:
        for recipe in result["recipes"]:
            if "response_time" in recipe:
                response_times.append(recipe["response_time"])
            if "quality_score" in recipe:
                quality_scores.append(recipe["quality_score"])

    avg_response_time = statistics.mean(response_times) if response_times else 0
    avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0

    # Populate benchmark results
    benchmark_results["overall"] = {
        "total_time": total_time,
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "avg_quality_score": avg_quality_score,
        "total_recipes": len(response_times),
        "queries_processed": len(TEST_QUERIES),
        "successful_queries": successful_queries
    }
    benchmark_results["queries"] = all_results

    # Generate and save report
    generate_report(benchmark_results)

    logger.info("Benchmark completed!")


def generate_report(results: Dict[str, Any]):
    """Generate a detailed benchmark report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": results["overall"],
        "detailed_results": []
    }

    # Add detailed analysis for each query
    for query_result in results["queries"]:
        query_summary = {
            "query": query_result["query"],
            "total_time": query_result["total_time"],
            "recipes_found": len(query_result["recipes"]),
            "errors": query_result["errors"],
            "avg_quality_score": query_result.get("avg_quality_score", 0),
            "recipes": []
        }

        # Add recipe details
        for recipe in query_result["recipes"]:
            recipe_detail = {
                "title": recipe.get("title", "Unknown"),
                "response_time": recipe.get("response_time", 0),
                "link": recipe.get("link", None),
                "quality_score": recipe.get("quality_score", 0),
                "evaluation": recipe.get("evaluation", {})
            }
            query_summary["recipes"].append(recipe_detail)

        report["detailed_results"].append(query_summary)

    # Save as JSON
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save as human-readable text report
    with open("benchmark_report.txt", "w") as f:
        f.write("==================================================\n")
        f.write("RECIPE BOT BENCHMARK REPORT\n")
        f.write(f"Generated: {report['timestamp']}\n")
        f.write("==================================================\n\n")

        # Write summary
        f.write("SUMMARY\n")
        f.write("-------\n")
        f.write(f"Total benchmark time: {report['summary']['total_time']:.2f} seconds\n")
        f.write(f"Success rate: {report['summary']['success_rate'] * 100:.1f}%\n")
        f.write(f"Average response time: {report['summary']['avg_response_time']:.2f} seconds\n")
        f.write(f"Average quality score: {report['summary']['avg_quality_score']:.1f}/10\n")
        f.write(f"Total recipes evaluated: {report['summary']['total_recipes']}\n")
        f.write(f"Queries processed: {report['summary']['queries_processed']}\n")
        f.write(f"Successful queries: {report['summary']['successful_queries']}\n\n")

        # Write detailed results
        f.write("DETAILED RESULTS\n")
        f.write("----------------\n\n")

        for i, query_result in enumerate(report["detailed_results"]):
            f.write(f"Query {i + 1}: {query_result['query']}\n")
            f.write(f"  Processing time: {query_result['total_time']:.2f} seconds\n")
            f.write(f"  Recipes found: {query_result['recipes_found']}\n")

            if query_result['errors']:
                f.write("  Errors encountered:\n")
                for error in query_result['errors']:
                    f.write(f"    - {error}\n")

            if query_result['recipes']:
                f.write(f"  Average quality score: {query_result['avg_quality_score']:.1f}/10\n")
                f.write("  Recipes:\n")

                for j, recipe in enumerate(query_result['recipes']):
                    f.write(f"    {j + 1}. {recipe['title']}\n")
                    f.write(f"       Response time: {recipe['response_time']:.2f} seconds\n")
                    f.write(f"       Quality score: {recipe['quality_score']:.1f}/10\n")

                    if recipe['evaluation']:
                        eval_data = recipe['evaluation']
                        f.write(f"       Evaluation: {eval_data.get('explanation', 'No explanation provided')}\n")

                        if 'strengths' in eval_data and eval_data['strengths']:
                            f.write("       Strengths:\n")
                            for strength in eval_data['strengths']:
                                f.write(f"         - {strength}\n")

                        if 'weaknesses' in eval_data and eval_data['weaknesses']:
                            f.write("       Weaknesses:\n")
                            for weakness in eval_data['weaknesses']:
                                f.write(f"         - {weakness}\n")

            f.write("\n")

        f.write("\n==================================================\n")
        f.write("END OF REPORT\n")
        f.write("==================================================\n")

    logger.info(f"Benchmark reports saved as benchmark_report.json and benchmark_report.txt")


if __name__ == "__main__":
    logger.info("Starting benchmark...")
    asyncio.run(run_benchmark())