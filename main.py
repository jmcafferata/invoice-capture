from flask import Flask, render_template, request, jsonify, send_from_directory
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
import numpy as np
from openai.embeddings_utils import cosine_similarity
from pathlib import Path
import os
from datetime import datetime
from io import StringIO
from ast import literal_eval
import pytz
import requests
timezone = pytz.timezone('America/Argentina/Buenos_Aires')
import json
import cv2
import pytesseract


app = Flask(__name__)


url = 'http://localhost:5502/'
# url = 'http://34.68.132.80:5502/'

# Show index.html
@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')

# Add data to json file
@app.route('/add_<dataName>', methods=['POST'])
def add_selection(dataName):
    try:
        data = request.get_json(force=True)
        with open(dataName+'.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

# Get data from json file
@app.route('/get_<data>', methods=['GET'])
def get_data(data):
    with open(data+'.json', 'r', encoding='utf-8') as f:
        data = f.read()
    return data

# Update data to json file
@app.route('/update_<dataName>/<idName>/<id>/<property>/<value>', methods=['POST'])
def update_data(dataName, idName,id, property, value):
    try:
        # Read the data
        with open(dataName+'.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Find the book in the data based on ISBN
        found = next((b for b in data if b[idName] == id), None)

        if found:
            # if the property is a number, keep the value as number, not string
            property_type = type(found[property])
            if property_type == int or property_type == float:
                value = property_type(value)
            # if it's boolean or a string that says true of false, convert the string to boolean
            elif property_type == bool or (property_type == str and (value == 'true' or value == 'false')):
                value = True if value == 'true' else False

            found[property] = value
        else:
            return jsonify({'status': 'error', 'error': 'Book not found in the data.'})

        # Write the updated data back to json file
        with open(dataName+'.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})
    
# Empty data from json file
@app.route('/empty_<dataName>', methods=['POST'])
def empty_data(dataName):
    with open(dataName+'.json', 'w', encoding='utf-8') as f:
        json.dump([], f)
    return jsonify({'status': 'success'})

# Upload image to server
@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400
        image = request.files['file']
        
        # Check if the file is empty
        if image.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file.'}), 400
        
        image_path = 'image.jpeg'
        image.save(image_path)
        
        extracted_text = ocr_image(image_path)
        items_json = extract_details(extracted_text)
 
        # replace selection.json with items_json
        with open('selection.json', 'w', encoding='utf-8') as f:
            json.dump(items_json, f, ensure_ascii=False, indent=4)
        return jsonify({'status': 'success', 'message': 'Image uploaded successfully.'})
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'error': str(e)})

import pdfplumber
import re

def ocr_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        # extract text from each page
        pages = []
        text = ''
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            pages.append(page)
            text += page.extract_text()
    return text

# Upload image to server
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400
        pdf = request.files['file']
        
        # Check if the file is empty
        if pdf.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file.'}), 400
        
        prompt = request.form['prompt']
        pdf_path = 'pdf.pdf'
        pdf.save(pdf_path)
        
        extracted_text = ocr_pdf(pdf_path)
        items_json = extract_details(extracted_text,prompt)

        return jsonify({'status': 'success', 'message': 'PDF uploaded successfully.', 'items': items_json})
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'error': str(e)})


# OCR uploaded image
def ocr_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale (this can improve OCR accuracy)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract text from the image using pytesseract
    extracted_text = pytesseract.image_to_string(gray)
    print("Extracted text: ", extracted_text)
    
    return extracted_text

# Extract details from OCR text
def extract_details(text,prompt):

    #get openai api key from openai_api_key.txt
    with open('openai_api_key.txt', 'r') as f:
        openai.api_key = f.read()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":prompt},
            {"role":"user","content":text},
            {"role":"assistant","content":"here's the array:\n"}
        ])
    response_text = response.choices[0].message.content
    # json loads the string into a dictionary
    print(response_text)
    # get only the text from the first '[' to the last ']'
    response_text = response_text[response_text.find('['):response_text.rfind(']')+1]
    response_json = json.loads(response_text)

    print(response_json)
    return response_json

    



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5502, debug=True)