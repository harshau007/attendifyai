import os
import base64
from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types
import openai
import re

load_dotenv()  # Load environment variables from .env file if it exists

def analyze_lab_image(image_path, api_type="chatgpt"):
    """
    Analyzes a lab CCTV image to extract timestamp, lab name, and headcount using either ChatGPT or Gemini API.

    Args:
        image_path (str): Path to the image file.
        api_type (str): API to use, either "chatgpt" or "gemini". Defaults to "chatgpt".

    Returns:
        dict: A dictionary containing 'timestamp', 'lab_name', and 'headcount',
              or an error message if analysis fails.
    """

    if api_type.lower() not in ["chatgpt", "gemini"]:
        return {"error": "Invalid API type. Choose 'chatgpt' or 'gemini'."}

    try:
        # --- API Key Handling ---
        if api_type.lower() == "chatgpt":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"error": "OPENAI_API_KEY environment variable not set. Please set your OpenAI API key."}
            client = openai.OpenAI(api_key=api_key)
            model_name = "gpt-4-vision-preview"  # GPT-4 Vision model

        elif api_type.lower() == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return {"error": "GEMINI_API_KEY environment variable not set. Please set your Gemini API key."}
            client = genai.Client(api_key=api_key)
            model_name = "gemini-2.0-flash"  # Gemini model you want to use

        # --- Image Loading ---
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            return {"error": f"File not found: {image_path}"}

        # --- Prompt Formulation ---
        prompt_text = """
        Analyze the CCTV image of a lab and extract the following information:
        1. Timestamp: Parse and extract the timestamp information visible in the image.
        2. Lab Name: Identify and extract the lab name or identifier visible in the image.
        3. Headcount: Count the number of people present in the lab in the image.

        Return the information in a tabular format as a markdown table, with columns "Category" and "Value".
        If you cannot find any information, please put "Not found" as value in the table.
        """

        # --- API Call ---
        if api_type.lower() == "chatgpt":
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": prompt_text,
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64,{base64.b64encode(image.tobytes()).decode()}"
                }],
                max_tokens=300,
            )
            api_response_text = response.choices[0].message.content

        elif api_type.lower() == "gemini":
            # Send image and prompt directly to Gemini API
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt_text, image]  # Send image object directly
            )
            api_response_text = response.text

        # --- Response Parsing and Table Extraction ---
        timestamp = "Not found"
        lab_name = "Not found"
        headcount = "Not found"

        # Use regex to find markdown table rows and extract values
        table_rows = re.findall(r'\| (Category) \| (Value) \||\n\| :--- \| :--- \||\n(?:\| ([^\|]+) \| ([^\|]+) \|(?:\n|$))?', api_response_text)

        if table_rows and len(table_rows) > 0 :  # Check if table is found
            for row in table_rows:
                if row[2].strip().lower() == "timestamp":
                    timestamp = row[3].strip()
                elif row[2].strip().lower() == "lab name":
                    lab_name = row[3].strip()
                elif row[2].strip().lower() == "headcount":
                    headcount = row[3].strip()

        return {
            "timestamp": timestamp,
            "lab_name": lab_name,
            "headcount": headcount
        }

    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

def process_images_in_folder(folder_path, api_type="gemini"):
    """
    Process all images in a folder and return consolidated results.
    """
    results = []
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)

        # Only process image files (you can modify this if you need other formats)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {image_filename}")
            result = analyze_lab_image(image_path, api_type)
            results.append({"image": image_filename, **result})
            print(f"Processed {image_filename}: {result}")

    return results

if __name__ == "__main__":
    image_folder_path = "images"  # Folder containing images
    api_to_use = "gemini"  # Choose either "chatgpt" or "gemini"

    results = process_images_in_folder(image_folder_path, api_to_use)

    # Print results in a table format
    print("\nConsolidated Analysis Result:")
    print("| Image Name        | Timestamp               | Lab Name                | Headcount              |")
    print("|-------------------|-------------------------|-------------------------|------------------------|")
    
    for result in results:
        print(f"| {result['image']} | {result.get('timestamp', 'Not found')} | {result.get('lab_name', 'Not found')} | {result.get('headcount', 'Not found')} |")
