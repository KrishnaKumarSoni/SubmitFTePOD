from flask import Flask, request, render_template, jsonify
import os
import openai
from dotenv import load_dotenv
import uuid
import logging
# import google.generativeai as genai
import json
import base64

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# genai.configure(api_key=os.environ['GEMINI_API_KEY'])

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

    logging.info(f"Sending image to {model_choice}")

    if model_choice == "openai":
        # Read the image and encode it in base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        image_data_url = f"data:image/jpeg;base64,{base64_image}"

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        print(response)
        return response.choices[0].message.content

    # elif model_choice == "gemini":
    #     # Upload the image using Gemini's API
    #     uploaded_file = genai.upload_file(image_path)
        
    #     # Choose the Gemini model (update if needed)
    #     model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    #     result = model.generate_content([uploaded_file, prompt])
    #     candidate = result.candidates[0]

    #     if hasattr(candidate, 'output') and candidate.output:
    #         return candidate.output
    #     else:
    #         logging.warning(f"No output generated. Finish reason: {candidate.finish_reason}")
    #         return f"No output generated. Finish reason: {candidate.finish_reason}"

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    phone = request.args.get('phone')
    model_choice = request.args.get('model')
    driver_name = driver_data.get(phone, "Driver")

    if request.method == "POST":
        image = request.files['image']
        if image:
            # Ensure temp_uploads directory exists
            temp_dir = "/tmp/temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            # Save the uploaded image to a temporary file
            temp_filename = f"{uuid.uuid4().hex}_{image.filename}"
            image_path = os.path.join(temp_dir, temp_filename)
            image.save(image_path)
            logging.info(f"Saved uploaded image to temporary path: {image_path}")
            
            # For OpenAI, generate image_url as base64 data URI
            if model_choice == "openai":
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{base64_image}"
            else:
                image_url = None  # Not needed for Gemini

            # Extract receiver details using the chosen AI model
            receiver_details = extract_receiver_details(image_url, image_path, model_choice)

            # Parse receiver_details to extract Name and Phone number
            name = ''
            phone_number = ''
            lines = receiver_details.split('\n')
            for line in lines:
                if line.lower().startswith('name:'):
                    name = line[len('Name:'):].strip()
                elif line.lower().startswith('phone number:'):
                    phone_number = line[len('Phone number:'):].strip()

            if os.path.exists(image_path):
                os.remove(image_path)
                logging.info(f"Deleted temporary file: {image_path}")

            return render_template("success.html", name=name, phone_number=phone_number)
    return render_template("upload.html", driver_name=driver_name)

@app.route("/submit-details", methods=["POST"])
def submit_details():
    name = request.form.get('name')
    phone_number = request.form.get('phone_number')
    # For prototype, we don't need to process the data further.
    return render_template("submission_success.html", name=name, phone_number=phone_number)

if __name__ == "__main__":
    app.run(debug=True)
