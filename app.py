from flask import Flask, request, render_template, jsonify
from google.cloud import storage
import os
import openai
from dotenv import load_dotenv
import uuid
import logging
import google.generativeai as genai
import json

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Google Cloud credentials (ensure the JSON key is placed in your project directory)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "submitftepod-ce04d6b2947b.json"
credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')

with open("service_account.json", "w") as file:
    file.write(credentials)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Hardcoded driver details for testing
driver_data = {
    "9999999999": "John Doe",
    "8888888888": "Jane Smith"
}

def extract_receiver_details(image_url, image_path, model_choice):
    prompt = ("Please carefully analyze the attached image. Your objective is to find the Receiver / Consignee's name "
              "and their phone number. I will provide you an image of the invoice. Your task is to check whether in the first "
              "place it contains the name and the phone number of the consignor or not. Only write what is present and "
              "carefully remember to report what is missing out of the details we want in the output (name and phone number "
              "of receiver / consignee). Do not make up the name or the phone number if it is not present. It is important "
              "to report what's missing and also important to show what's present out of the two data points we want. Take a "
              "deep breath and think step by step."
              "Be careful. Do not mistake driver or the consignor / sender details as the receiver / consignee details. Ensure you correctly recognise what's consignee detail and then report only that. Do not report wrong details."
              "Response format: "
              "Name: "
              "Phone number: ")

    logging.info(f"Sending image URL to {model_choice}: {image_url}")

    if model_choice == "openai":
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
                }
            ],
            max_tokens=1000,
    )
        print(response)
        return response.choices[0].message.content

    elif model_choice == "gemini":
        # Upload the image using Gemini's API
        uploaded_file = genai.upload_file(image_path)
        
        # Choose the Gemini model (update if needed)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        result = model.generate_content([uploaded_file, prompt])
        candidate = result.candidates[0]

        if hasattr(candidate, 'output') and candidate.output:
            return candidate.output
        else:
            logging.warning(f"No output generated. Finish reason: {candidate.finish_reason}")
            return f"No output generated. Finish reason: {candidate.finish_reason}"




def generate_signed_url(phone, filename):
    bucket_name = 'epod-uploads'
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{phone}/{filename}")

    # Generate a signed URL valid for 1 hour (3600 seconds)
    signed_url = blob.generate_signed_url(version="v4", expiration=3600)
    return signed_url

def upload_image_to_gcp(image_path, phone):
    bucket_name = 'epod-uploads'
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    unique_filename = f"POD_{uuid.uuid4().hex}_{os.path.basename(image_path)}"
    blob = bucket.blob(f"{phone}/{unique_filename}")
    
    logging.info(f"Uploading image to bucket: {bucket_name}, with filename: {unique_filename}")
    
    # Upload directly from the file system
    blob.upload_from_filename(image_path, content_type="image/jpeg")  # Adjust content type as needed
    
    signed_url = generate_signed_url(phone, unique_filename)
    logging.info(f"Generated signed URL: {signed_url}")

    return signed_url




@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    phone = request.args.get('phone')
    model_choice = request.args.get('model')
    driver_name = driver_data.get(phone, "Driver")

    if request.method == "POST":
        image = request.files['image']
        if image:
            # Ensure temp_uploads directory exists
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            # Save the uploaded image to a temporary file
            temp_filename = f"{uuid.uuid4().hex}_{image.filename}"
            image_path = os.path.join(temp_dir, temp_filename)
            image.save(image_path)
            logging.info(f"Saved uploaded image to temporary path: {image_path}")
            

            image_url = upload_image_to_gcp(image_path, phone)

            # Extract receiver details using OpenAI's API
            receiver_details = extract_receiver_details(image_url, image_path, model_choice)
            if os.path.exists(image_path):
                os.remove(image_path)
                logging.info(f"Deleted temporary file: {image_path}")
            return render_template("success.html", image_url=image_url, receiver_details=receiver_details)

    return render_template("upload.html", driver_name=driver_name)


# API endpoint to check image upload status
@app.route("/check-upload", methods=["GET"])
def check_upload():
    phone = request.args.get('phone')
    bucket_name = 'epod-uploads'
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=f"{phone}/"))

    if blobs:
        signed_url = generate_signed_url(phone, blobs[0].name)
        return jsonify({"status": "success", "message": "Image found", "image_url": signed_url})
    else:
        return jsonify({"status": "fail", "message": "No image found"})

