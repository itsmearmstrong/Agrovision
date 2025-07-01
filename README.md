# Agrovision
Smart strawberry crop monitoring using YOLO NAS, IoT, and GIS. Detects ripe fruits, diseases, and weeds in real-time via a rover with video capture, remote control, and geospatial mapping to support precision agriculture.

AGROVISION: Smart Strawberry Crop Monitoring System

AGROVISION is an intelligent crop monitoring system designed to support precision agriculture in strawberry farms. It uses a mobile rover equipped with a camera and controlled via the Blynk IoT platform to capture real-time video of crop rows. The captured frames are analyzed using a YOLO NAS object detection model to identify:

🍓 Ripe strawberries for optimized harvesting

🦠 Plant diseases for early diagnosis and treatment

🌱 Weeds for selective removal

Detected outputs are geotagged using GPS and visualized in QGIS, enabling farmers to make location-specific decisions. The system reduces manual labor, improves harvest accuracy, and enhances overall crop health management. Designed for scalability, AGROVISION is a low-cost, real-time, and modular solution adaptable to other crops with minor changes.

🛠️ Technologies Used
YOLO NAS (You Only Look Once - Neural Architecture Search)

Python, OpenCV, TensorFlow/PyTorch

NodeMCU ESP8266

Blynk IoT Platform

QGIS (Geographic Information System)

Custom object detection datasets from Kaggle and field images

🌾 Objective
To automate key agricultural tasks—ripeness detection, disease monitoring, and weed identification—into a single, accessible system for small to mid-sized strawberry farms, enhancing precision farming practices.
