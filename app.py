# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from google.cloud import vision
import os
import io
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize Google Cloud Vision client
client = vision.ImageAnnotatorClient()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Dictionary of strawberry diseases with their solutions
DISEASE_SOLUTIONS = {
    "powdery mildew": {
        "description": "A fungal disease that appears as white powdery spots on leaves and fruits.",
        "solution": "Apply sulfur-based fungicides early in the season. Increase air circulation by proper plant spacing. Remove and destroy infected plant parts."
    },
    "leaf spot": {
        "description": "Characterized by small purple or red spots that develop gray or white centers on leaves.",
        "solution": "Apply copper-based fungicides. Practice crop rotation. Remove infected leaves promptly. Avoid overhead watering to keep foliage dry."
    },
    "gray mold": {
        "description": "Also known as Botrytis fruit rot, causing fuzzy gray growth on fruits and flowers.",
        "solution": "Improve air circulation. Remove infected fruits and plant debris. Apply fungicides during flowering. Harvest ripe berries promptly."
    },
    "angular leaf spot": {
        "description": "Bacterial disease causing water-soaked spots on leaves that later turn reddish-brown.",
        "solution": "Use certified disease-free plants. Apply copper-based bactericides. Avoid overhead irrigation. Remove infected plants."
    },
    "anthracnose fruit rot": {
        "description": "A fungal disease causing dark, sunken lesions on fruits and can affect runners, crowns, and petioles.",
        "solution": "Apply fungicides preventatively. Use disease-free nursery stock. Remove infected plant material. Improve air circulation. Use plastic mulch to reduce splash dispersal."
    },
    "tip burn": {
        "description": "A physiological disorder causing browning and drying of leaf margins, often due to calcium deficiency or environmental stress.",
        "solution": "Ensure adequate calcium supply. Maintain consistent soil moisture. Avoid high salt levels in soil. Protect plants from environmental extremes."
    }
}

# Additional disease indicators and related terms to help with detection
DISEASE_INDICATORS = {
    "powdery mildew": ["powdery", "mildew", "white powder", "white spot", "whitish", "fungal"],
    "leaf spot": [
        "leaf spot", "purple spot", "red spot", "brown spot", "circular spot", "necrotic lesion", "tan spot", "yellow halo", "leaf lesion", "spot", "lesion", "center", "necrotic"
    ],
    "gray mold": ["gray mold", "botrytis", "fuzzy", "gray growth", "rot", "botrytis fruit rot"],
    "angular leaf spot": ["angular", "bacterial", "water-soaked", "reddish-brown", "leaf spot"],
    "anthracnose fruit rot": ["anthracnose", "fruit rot", "dark lesion", "sunken", "lesion", "black spot"],
    "tip burn": ["tip burn", "leaf margin", "brown edge", "calcium", "leaf edge", "dry edge", "browning"]
}

# No terms to exclude now
EXCLUDED_DISEASE_TERMS = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

class ImageAnalyzer:
    def analyze_image(self, image_path):
        # Read image file
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        # Read image for processing with OpenCV
        image_cv = cv2.imread(image_path)
        height, width = image_cv.shape[:2]

        # Create image object for Vision API
        image = vision.Image(content=content)
        
        # Perform detection using multiple features
        context = vision.ImageContext(crop_hints_params=vision.CropHintsParams(aspect_ratios=[1.0]))
        
        # Get annotations with more features for better disease detection
        response = client.annotate_image({
            'image': image,
            'features': [
                {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
                {'type_': vision.Feature.Type.LABEL_DETECTION, 'max_results': 50},
                {'type_': vision.Feature.Type.WEB_DETECTION, 'max_results': 50},
                {'type_': vision.Feature.Type.IMAGE_PROPERTIES},
                {'type_': vision.Feature.Type.CROP_HINTS},
                {'type_': vision.Feature.Type.TEXT_DETECTION},
            ],
            'image_context': context,
        })
        
        # Process results and draw bounding boxes
        results, processed_image_path = self._process_results(response, image_cv, width, height, image_path)
        
        return results, processed_image_path
    
    def _process_results(self, response, image, width, height, original_image_path):
        results = {
            'ripe_strawberries': [],
            'weeds': [],
            'diseases': []
        }
        
        # Process object detection for strawberries and draw boxes
        strawberry_found = False
        for object_ in response.localized_object_annotations:
            if object_.name.lower() in ['strawberry', 'fruit'] and object_.score >= 0.3:  # Lowered threshold, allow 'fruit'
                strawberry_found = True
                # Extract normalized vertices
                vertices = [(vertex.x * width, vertex.y * height) for vertex in object_.bounding_poly.normalized_vertices]
                # Convert to pixel coordinates and integer type
                box = np.array(vertices, np.int32)
                # Draw rectangle (green for ripe strawberries)
                cv2.polylines(image, [box], True, (0, 255, 0), 3)
                # Add label with confidence
                label = f"{object_.name}: {object_.score:.2f}"
                label_position = (int(vertices[0][0]), int(vertices[0][1] - 10))
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Add to results
                results['ripe_strawberries'].append({
                    'confidence': object_.score,
                    'name': object_.name,
                    'box': [[vertex.x, vertex.y] for vertex in object_.bounding_poly.normalized_vertices]
                })
        # Fallback: if no strawberries detected by object localization, check label annotations
        if not strawberry_found:
            for label in response.label_annotations:
                if label.description.lower() in ['strawberry', 'fruit']:
                    # Add a low-confidence detection
                    results['ripe_strawberries'].append({
                        'confidence': label.score if hasattr(label, 'score') else 0.2,
                        'name': label.description,
                        'box': []
                    })
                    break
        
        # Collect all descriptions, labels, and text for disease detection
        all_descriptions = set()
        
        # Process labels for disease and weed detection
        for label in response.label_annotations:
            all_descriptions.add(label.description.lower())
        
        # Add web detection results for better disease recognition
        if response.web_detection:
            for entity in response.web_detection.web_entities:
                all_descriptions.add(entity.description.lower())
                
            # Add web labels
            if response.web_detection.best_guess_labels:
                for label in response.web_detection.best_guess_labels:
                    all_descriptions.add(label.label.lower())
            
            # Add web labels from pages
            for page in response.web_detection.pages_with_matching_images:
                if page.page_title:
                    all_descriptions.add(page.page_title.lower())
        
        # Add OCR text for possible plant disease mentions
        if response.text_annotations and len(response.text_annotations) > 0:
            all_descriptions.add(response.text_annotations[0].description.lower())
            
        # Check image properties for common disease colors (reddish, yellowish, etc.)
        if response.image_properties_annotation:
            dominant_colors = response.image_properties_annotation.dominant_colors.colors
            for color in dominant_colors[:5]:  # Check top 5 dominant colors
                # Check for powdery white colors (specific to powdery mildew)
                if (color.color.red > 220 and color.color.green > 220 and color.color.blue > 220 and
                    abs(color.color.red - color.color.green) < 20 and 
                    abs(color.color.green - color.color.blue) < 20):
                    all_descriptions.add("powdery white")
                    all_descriptions.add("powdery mildew")
                
                # Check for yellowish colors (could indicate disease)
                if color.color.red > 200 and color.color.green > 200 and color.color.blue < 100:
                    all_descriptions.add("yellow discoloration")
                
                # Check for reddish-brown colors (could indicate leaf spot or scorch)
                if color.color.red > 150 and color.color.green < 100 and color.color.blue < 100:
                    all_descriptions.add("reddish discoloration")
                
                # Check for gray colors (specific to gray mold)
                if (color.color.red > 120 and color.color.red < 160 and 
                    color.color.green > 120 and color.color.green < 160 and 
                    color.color.blue > 120 and color.color.blue < 160 and
                    abs(color.color.red - color.color.green) < 15 and 
                    abs(color.color.green - color.color.blue) < 15 and
                    color.score > 0.3):  # Only consider significant color presence
                    all_descriptions.add("grayish")
                    all_descriptions.add("gray mold")

        # Check for specific strawberry plant keywords
        plant_keywords = ["strawberry", "strawberries", "plant", "leaf", "leaves", "fruit", "garden"]
        has_plant_context = any(keyword in " ".join(all_descriptions) for keyword in plant_keywords)
        
        # Disease detection confidence will be higher if plant context is detected
        base_confidence = 0.8 if has_plant_context else 0.6
        
        # Check for specific diseases with improved matching
        detected_diseases = []
        
        # First, look for direct disease name matches
        for term in all_descriptions:
            # Skip excluded disease terms
            if any(excluded_term in term for excluded_term in EXCLUDED_DISEASE_TERMS):
                continue
                
            # Check for specific strawberry diseases by name
            for disease_name, info in DISEASE_SOLUTIONS.items():
                if disease_name in term:
                    # Special handling for powdery mildew vs gray mold
                    if disease_name == "powdery mildew" and "powdery white" in all_descriptions:
                        detected_disease = {
                            'name': disease_name.title(),
                            'description': info['description'],
                            'solution': info['solution'],
                            'confidence': base_confidence + 0.2  # Higher confidence for powdery mildew with white color
                        }
                    elif disease_name == "gray mold":
                        # More strict criteria for gray mold
                        gray_mold_indicators = ["grayish", "gray mold", "botrytis", "fuzzy", "gray growth"]
                        if (("grayish" in all_descriptions and 
                             any(indicator in " ".join(all_descriptions) for indicator in gray_mold_indicators) and
                             has_plant_context)):  # Require plant context for gray mold
                            detected_disease = {
                                'name': disease_name.title(),
                                'description': info['description'],
                                'solution': info['solution'],
                                'confidence': base_confidence + 0.2  # Higher confidence for gray mold with multiple indicators
                            }
                        else:
                            # Skip if criteria not met
                            continue
                    else:
                        detected_disease = {
                            'name': disease_name.title(),
                            'description': info['description'],
                            'solution': info['solution'],
                            'confidence': base_confidence + 0.1
                        }
                    
                    # Don't add duplicates
                    if not any(d['name'].lower() == disease_name for d in detected_diseases):
                        detected_diseases.append(detected_disease)
        
        # Next, check for disease indicators and related terms
        for term in all_descriptions:
            for disease_name, indicators in DISEASE_INDICATORS.items():
                # Skip if we already found this disease by name
                if any(d['name'].lower() == disease_name for d in detected_diseases):
                    continue
                # Check for indicators of this disease
                indicator_matches = [indicator for indicator in indicators if indicator in term]
                if indicator_matches:
                    confidence_boost = 0.1 * min(len(indicator_matches), 3)  # Boost confidence if multiple indicators
                    detected_disease = {
                        'name': disease_name.title(),
                        'description': DISEASE_SOLUTIONS[disease_name]['description'],
                        'solution': DISEASE_SOLUTIONS[disease_name]['solution'],
                        'confidence': base_confidence + confidence_boost if disease_name == 'leaf spot' else base_confidence
                    }
                    detected_diseases.append(detected_disease)
                    break  # Stop checking indicators once we found one
        
        # Check for general disease indicators if no specific disease found
        if not detected_diseases:
            general_disease_indicators = [
                'disease', 'blight', 'mold', 'mildew', 'rot', 'spot', 'wilt', 'scorch',
                'discoloration', 'yellowing', 'browning', 'wilting', 'spots', 'lesion',
                'infection', 'fungal', 'bacterial', 'pathogen'
            ]
            
            # If we detect general disease indicators but no specific disease
            for term in all_descriptions:
                if any(indicator in term for indicator in general_disease_indicators):
                    # Look for color indicators to narrow down
                    if 'white' in term or 'powdery' in term:
                        disease = "powdery mildew"
                    elif 'gray' in term or 'grey' in term:
                        disease = "gray mold"
                    elif 'red' in term or 'purple' in term or 'spot' in term:
                        disease = "leaf spot"
                    elif 'yellow' in term or 'wilt' in term:
                        disease = "verticillium wilt"
                    else:
                        # Skip if we can't narrow it down
                        continue
                        
                    detected_disease = {
                        'name': disease.title(),
                        'description': DISEASE_SOLUTIONS[disease]['description'],
                        'solution': DISEASE_SOLUTIONS[disease]['solution'],
                        'confidence': base_confidence - 0.1  # Lower confidence for inferred matches
                    }
                    
                    # Don't add duplicates
                    if not any(d['name'].lower() == disease for d in detected_diseases):
                        detected_diseases.append(detected_disease)
        
        # Check for weeds with improved detection
        weed_indicators = ['weed', 'grass', 'invasive', 'unwanted plant', 'crabgrass', 'dandelion', 'nettle']
        for term in all_descriptions:
            if any(indicator in term for indicator in weed_indicators):
                # Try to identify the specific weed type if possible
                weed_type = "Unknown weed"
                if 'crabgrass' in term:
                    weed_type = "Crabgrass"
                elif 'dandelion' in term:
                    weed_type = "Dandelion"
                elif 'nettle' in term:
                    weed_type = "Nettle"
                elif 'grass' in term:
                    weed_type = "Grass weed"
                
                results['weeds'].append({
                    'label': weed_type,
                    'confidence': 0.7  # Default confidence
                })
        
        # Add detected diseases to results
        results['diseases'] = detected_diseases
        
        # Save processed image
        filename = f"{uuid.uuid4()}.jpg"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        cv2.imwrite(processed_path, image)
        
        return results, filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            analyzer = ImageAnalyzer()
            results, processed_image = analyzer.analyze_image(filepath)
            
            # Clean up original uploaded file
            os.remove(filepath)
            
            # Add processed image path to results
            results['processed_image'] = processed_image
            
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_video_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            # Open video file
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(frame_count / fps) if fps > 0 else frame_count
            analyzer = ImageAnalyzer()
            processed_frames = []
            frame_idx = 0
            sec = 0
            while cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # Convert frame to RGB and save with high quality
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                temp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{frame_idx}.jpg")
                # Save with high JPEG quality
                cv2.imwrite(temp_img_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                results, processed_image = analyzer.analyze_image(temp_img_path)
                # Only keep frames with strawberries or diseases
                if (results['ripe_strawberries'] and len(results['ripe_strawberries']) > 0) or (results['diseases'] and len(results['diseases']) > 0):
                    processed_frames.append({
                        'frame_index': frame_idx,
                        'timestamp': sec,
                        'results': results,
                        'processed_image': processed_image
                    })
                os.remove(temp_img_path)
                frame_idx += 1
                sec += 1
                if sec > duration:
                    break
            cap.release()
            # Ensure file is released before deleting (Windows fix)
            time.sleep(0.1)
            os.remove(filepath)
            return jsonify({'frames': processed_frames})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
