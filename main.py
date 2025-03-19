import os
import io
import re
import time
import gc
import threading
import concurrent.futures 
import datetime
import shutil

import cv2
import numpy as np
import torch
import torchvision.ops as ops
import tensorflow as tf
import tensorflow_hub as hub
import easyocr
import numpy as np
import datetime
import time

# ---------------------------
# SMTPLib for Emailing
# ---------------------------

import smtplib
from email.message import EmailMessage
import ssl

# ---------------------------
# Google Drive API Imports (for script mode)
# ---------------------------
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
except ImportError:
    # Not needed in Colab mode if not running as a script.
    pass

# ---------------------------
# ReportLab for PDF Report
# ---------------------------
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as RLImage
from reportlab.lib import colors

# ---------------------------
# Global Variables for Models and Concurrency Locks
# ---------------------------
sr = None
yolo_model = None
effdet_model = None

yolo_lock = threading.Lock()
effdet_lock = threading.Lock()
sr_lock = threading.Lock()

# ---------------------------
# Model Initialization Function
# ---------------------------
def init_models():
    global sr, yolo_model, effdet_model
    # Super Resolution Model (requires EDSR_x4.pb to be in the working directory)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x4.pb")
    sr.setModel("edsr", 4)  # Using a scale factor of 4

    # YOLO Model using ultralytics (make sure yolov8x.pt is available)
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install ultralytics (pip install ultralytics) to run YOLO detection.")
    yolo_model = YOLO('yolov8x.pt')

    # EfficientDet model from TensorFlow Hub
    effdet_model = hub.load('https://tfhub.dev/tensorflow/efficientdet/d7/1')
    print("Models have been initialized.")

# ---------------------------
# Google Drive Functions (for script mode)
# ---------------------------
def list_images(folder_id, drive_service):
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files_list = results.get('files', [])
    return files_list

def download_image(file_id, file_name, drive_service):
    folder_path = "./images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    request = drive_service.files().get_media(fileId=file_id)
    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    print(f"✅ Downloaded: {file_name} → {file_path}")
    return file_path

# ---------------------------
# OCR Function to Extract Lab Number
# ---------------------------
def process_ocr(image_path):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Could not load image: {image_path}")
        return "N/A"
    reader = easyocr.Reader(['en'], gpu=False)
    ocr_results = reader.readtext(image_cv, detail=0)
    combined_text = " ".join(ocr_results)
    # Look for three-digit numbers (adjust regex if needed)
    lab_numbers = re.findall(r'\b\d{3}\b', combined_text)
    if lab_numbers:
        # Use the last found lab number
        lab_number = lab_numbers[-1]
        return lab_number
    else:
        return "N/A"

# ---------------------------
# Detection Functions: YOLO & EfficientDet on Image Patches
# ---------------------------
def detect_yolo_multiscale(patch, yolo_model, scales=[0.8, 1.0, 1.2], conf_threshold=0.5):
    boxes = []
    scores = []
    for scale in scales:
        scaled_patch = cv2.resize(patch, None, fx=scale, fy=scale)
        with yolo_lock:
            results = yolo_model(scaled_patch, conf=conf_threshold)
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            if yolo_model.names[cls] == "person":
                x, y, w, h = box.xywh[0].cpu().numpy()
                x, y, w, h = x/scale, y/scale, w/scale, h/scale
                x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
                boxes.append([x1, y1, x2, y2])
                scores.append(box.conf[0].item())
    if boxes:
        return np.array(boxes), np.array(scores)
    else:
        return np.empty((0, 4)), np.empty((0,))

def detect_effdet_patch(patch, effdet_model, conf_threshold=0.5):
    patch_h, patch_w = patch.shape[:2]
    eff_image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    input_image = tf.convert_to_tensor(eff_image)
    input_image = tf.expand_dims(input_image, axis=0)
    with effdet_lock:
        eff_results = effdet_model(input_image)
    boxes_tf = eff_results['detection_boxes'].numpy()[0]
    scores_tf = eff_results['detection_scores'].numpy()[0]
    classes_tf = eff_results['detection_classes'].numpy()[0].astype(np.int32)
    person_mask = np.logical_and(classes_tf == 1, scores_tf >= conf_threshold)
    eff_boxes_norm = boxes_tf[person_mask]
    eff_scores = scores_tf[person_mask]
    eff_boxes = []
    for box in eff_boxes_norm:
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin * patch_w)
        y1 = int(ymin * patch_h)
        x2 = int(xmax * patch_w)
        y2 = int(ymax * patch_h)
        eff_boxes.append([x1, y1, x2, y2])
    if eff_boxes:
        return np.array(eff_boxes), np.array(eff_scores)
    else:
        return np.empty((0, 4)), np.empty((0,))

def detect_on_patch(patch, yolo_model, effdet_model, conf_threshold=0.5):
    yolo_boxes, yolo_scores = detect_yolo_multiscale(patch, yolo_model, conf_threshold=conf_threshold)
    eff_boxes, eff_scores = detect_effdet_patch(patch, effdet_model, conf_threshold=conf_threshold)

    if yolo_boxes.shape[0] == 0 and eff_boxes.shape[0] == 0:
        ensemble_boxes = np.empty((0, 4))
        ensemble_scores = np.empty((0,))
    else:
        if yolo_boxes.shape[0] > 0 and eff_boxes.shape[0] > 0:
            ensemble_boxes = np.vstack([yolo_boxes, eff_boxes])
            ensemble_scores = np.concatenate([yolo_scores, eff_scores])
        elif yolo_boxes.shape[0] > 0:
            ensemble_boxes = yolo_boxes
            ensemble_scores = yolo_scores
        else:
            ensemble_boxes = eff_boxes
            ensemble_scores = eff_scores

    if ensemble_boxes.shape[0] > 0:
        boxes_tensor = torch.tensor(ensemble_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(ensemble_scores, dtype=torch.float32)
        nms_indices = ops.nms(boxes_tensor, scores_tensor, 0.5)
        final_boxes = boxes_tensor[nms_indices].numpy()
        final_scores = scores_tensor[nms_indices].numpy()
    else:
        final_boxes = np.empty((0, 4))
        final_scores = np.empty((0,))
    return final_boxes, final_scores

# ---------------------------
# Full Image Processing
# ---------------------------
def process_file(fn):
    print(f"Processing file: {fn}")
    orig_image = cv2.imread(fn)
    if orig_image is None:
        print(f"Error reading {fn}")
        return fn, None, None, 0

    with sr_lock:
        enhanced_image = sr.upsample(orig_image)
    enh_h, enh_w = enhanced_image.shape[:2]
    mid_x, mid_y = enh_w // 2, enh_h // 2

    quadrants = {
        'top_left':    (enhanced_image[0:mid_y, 0:mid_x], 0, 0),
        'top_right':   (enhanced_image[0:mid_y, mid_x:enh_w], mid_x, 0),
        'bottom_left': (enhanced_image[mid_y:enh_h, 0:mid_x], 0, mid_y),
        'bottom_right':(enhanced_image[mid_y:enh_h, mid_x:enh_w], mid_x, mid_y)
    }

    total_count = 0
    all_boxes_global = []
    all_scores_global = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as quad_executor:
        future_to_quad = {}
        for quad_name, (quad_img, offset_x, offset_y) in quadrants.items():
            future = quad_executor.submit(detect_on_patch, quad_img, yolo_model, effdet_model, 0.5)
            future_to_quad[future] = (quad_name, offset_x, offset_y)
        for future in concurrent.futures.as_completed(future_to_quad):
            quad_name, offset_x, offset_y = future_to_quad[future]
            quad_boxes, quad_scores = future.result()
            count_quad = quad_boxes.shape[0]
            total_count += count_quad
            for box in quad_boxes:
                x1, y1, x2, y2 = box
                all_boxes_global.append([x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y])
            for score in quad_scores:
                all_scores_global.append(score)
            print(f"{quad_name}: Detected {count_quad} persons.")

    print(f"Total persons detected in {fn}: {total_count}")

    output_image = enhanced_image.copy()
    for i, box in enumerate(all_boxes_global):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person: {all_scores_global[i]:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    del orig_image, enhanced_image, quadrants, all_boxes_global, all_scores_global
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fn, output_image, total_count, None

# ---------------------------
# PDF Report Generation Function
# ---------------------------
def create_pdf(data, filename="output.pdf", header_image=None):
    margin = 10
    document = SimpleDocTemplate(filename, pagesize=letter,
                                 rightMargin=margin, leftMargin=margin,
                                 topMargin=margin, bottomMargin=margin)
    elements = []

    # Add header image if exists
    if header_image:
        img = RLImage(header_image, width=letter[0]-2*margin, height=100)
        elements.append(img)

    # Add generated date
    generated_date = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")
    date_table = Table([[f"Generated on: {generated_date}"]],
                      colWidths=[document.pagesize[0]-2*margin])
    date_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.grey),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
    ]))
    elements.append(date_table)

    # Add main data table with the provided lab results data.
    table_data = [["Lab No", "Head Count", "Time"]]
    table_data.extend(data)
    table = Table(table_data, colWidths=[document.pagesize[0]*0.3]*3)
    table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,0), (-1,0), (0.7, 0.7, 0.7)),
        ('GRID', (0,0), (-1,-1), 0.5, (0,0,0)),
        ('LINEBELOW', (0,0), (-1,0), 1, (0,0,0))
    ]))
    elements.append(table)
    document.build(elements)
    print(f"PDF report created: {filename}")

# ---------------------------
# Email Sending Function
# ---------------------------
def send_email(sender_email, app_password, recipients, subject, plain_text, pdf_path):
    """
    Parameters:
      - sender_email: Your Gmail address.
      - app_password: Your Gmail app-specific password.
      - recipients: A list of recipient email addresses.
      - subject: Subject line of the email.
      - plain_text: Plain text version of the email content.
      - pdf_path: Path to the PDF file to attach.
    """
    html_content = """\
    <html>
      <head>
        <style>
          body {
            font-family: 'Helvetica', Arial, sans-serif;
            background-color: #f9f9f9;
            color: #333333;
            margin: 0;
            padding: 20px;
          }
          .container {
            background-color: #ffffff;
            margin: auto;
            padding: 20px;
            max-width: 600px;
            border: 1px solid #dddddd;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          }
          h1 {
            font-size: 24px;
            margin-bottom: 20px;
            color: #222222;
          }
          p {
            line-height: 1.6;
            font-size: 16px;
            margin-bottom: 20px;
          }
          .footer {
            margin-top: 20px;
            font-size: 12px;
            text-align: center;
            color: #777777;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Daily Lab Report</h1>
          <p>Please check the attachment for detailed information in the PDF report.</p>
          <div class="footer">
            &copy; 2025 TCET. All rights reserved.
          </div>
        </div>
      </body>
    </html>
    """
    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject
    
    # Set the plain text content as fallback
    msg.set_content(plain_text)
    
    # Add the HTML content as an alternative
    msg.add_alternative(html_content, subtype='html')
    
    # Attach the PDF file
    try:
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
            file_name = pdf_path.split("/")[-1]
        msg.add_attachment(pdf_data, maintype='application', subtype='pdf', filename=file_name)
    except FileNotFoundError:
        print("The specified PDF file was not found.")
        return
    
    # Connect to Gmail's SMTP server using SSL
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print("Email sent successfully!")
    except smtplib.SMTPException as e:
        print("An error occurred while sending the email:", e)

# ---------------------------
# Function to Extract Time from Filename
# ---------------------------
def extract_time_from_filename(filename):
    """
    Extracts and formats the time portion (HH:MM:SS) from an image filename.
    The expected pattern in the filename is YYYYMMDDHHMMSS.
    It checks that the year part (YYYY) is equal to or greater than the current year.
    
    For example, for a filename like "photo_20250213121339_extra.jpg",
    it will return "12:13:39" if the year (2025) is valid.
    """
    match = re.search(r'(\d{14})', filename)
    if match:
        date_time_str = match.group(1)
        current_year = datetime.datetime.now().year
        try:
            year_in_filename = int(date_time_str[:4])
        except ValueError:
            return None
        
        if year_in_filename >= current_year:
            time_str = date_time_str[8:]
            # Format the time string with colons (HH:MM:SS)
            formatted_time = ':'.join([time_str[i:i+2] for i in range(0, len(time_str), 2)])
            return formatted_time
    return None

# ---------------------------
# Cleanup Function: Upload PDF to Backup Folder & Delete Files/Folders
# ---------------------------
def cleanup(drive_service):
    # Rename the PDF file to include today's timestamp
    if os.path.exists("output.pdf"):
        new_filename = "report" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".pdf"
        os.rename("output.pdf", new_filename)
        extracted_time = extract_time_from_filename(new_filename)
        print("Extracted time from filename:", extracted_time)
        
        # Delete the local PDF file after uploading
        os.remove(new_filename)
        print(f"Deleted local file: {new_filename}")
    else:
        print("output.pdf does not exist.")

    # Delete folders './images' and './processed'
    folders = ['./images', './processed']
    for folder in folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)  # Recursively delete folder and all its contents
                print(f"Deleted folder: {folder}")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")
        else:
            print(f"Folder {folder} does not exist")

# ---------------------------
# Main Function
# ---------------------------
def run_script():
    SERVICE_ACCOUNT_FILE = 'credential.json'
    SCOPES = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)

    folder_id = "FOLDER_ID"
    files_list = list_images(folder_id, drive_service)
    if not files_list:
        print("No images found in the specified folder.")
        return

    init_models()
    lab_results = {}  # key: lab number, value: (output_image, total_count, time, output_filename)

    # Create a folder named 'processed' if it doesn't exist
    processed_folder = "processed"
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    for file in files_list:
        file_id = file['id']
        file_name = file['name']
        local_path = download_image(file_id, file_name, drive_service)

        lab_no = process_ocr(local_path)
        fn, output_image, total_count, _ = process_file(local_path)
        
        # Construct the path to save the processed image inside the 'processed' folder
        output_filename = os.path.join(processed_folder, "processed_" + file_name)
        
        # Save the image inside the 'processed' folder
        cv2.imwrite(output_filename, output_image)
        
        # Extract time from the input file's filename; if extraction fails, use the current time.
        extracted_time = extract_time_from_filename(file_name)
        if extracted_time is None:
            extracted_time = time.strftime("%I:%M %p")
        lab_results[lab_no] = (output_image, total_count, extracted_time, output_filename)

    results_data = []
    for lab_no, (output_image, total_count, extracted_time, output_filename) in lab_results.items():
        results_data.append([lab_no, total_count, extracted_time])

    header_img = "tcetLogo.jpg" if os.path.exists("tcetLogo.jpg") else None
    create_pdf(results_data, filename="output.pdf", header_image=header_img)
    send_email(
        sender_email="amxngc@gmail.com",
        app_password="APP_PASS",
        recipients=["amanupadhyay2004@gmail.com", "amir.kamal09@gmail.com", "dhirajkalwar57@gmail.com"],
        subject="Test Email with HTML and PDF Attachment",
        plain_text="This is the plain text fallback message.",
        pdf_path="output.pdf"
    )
    # After sending email, perform cleanup (upload PDF to backup folder & delete local files)
    cleanup(drive_service)

# ---------------------------
# Entry Point: Choose Mode Based on Environment
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()
    run_script()
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Execution time: {execution_time} min")