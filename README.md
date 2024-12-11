# Real-Time Object Detection and Database

This project is a Flask-based web application that performs real-time object detection using a YOLOv8 model and stores detection data in an SQLite database. The application streams live webcam feed, detects objects, and allows users to view and interact with detection records in real time.

---

## Features
- *Real-Time Object Detection*: Uses the YOLOv8 model to detect objects from a live webcam feed.
- *Database Integration*: Saves detection results (product, quantity, and timestamp) into an SQLite database.
- *Dynamic Data Updates*: Displays detection records on the same page, dynamically updated every second.
- *Web Controls*: Includes Play, Pause, and End controls for the webcam feed.
- *Time-Based Database Updates*: Ensures records are added to the database only if the last timestamp for a detected product is at least 20 seconds old.

---

## Requirements

### Software Requirements:
- Python 3.7+
- Flask
- OpenCV
- ultralytics (YOLOv8)
- SQLite

### Python Dependencies:
Install the required dependencies using pip:
bash
pip install flask opencv-python ultralytics


---

## Setup Instructions

### 1. Clone the Repository
bash
git clone https://github.com/yourusername/repository-name.git
cd repository-name


### 2. Prepare the YOLOv8 Model
- Place your trained YOLOv8 model file (best2.pt) in the project directory.

### 3. Run the Application
- Create a Procfile for deployment (if needed) with the following content:
  
  web: python app.py
  
- Run the application locally:
  bash
  python app.py
  
- Open your browser and navigate to http://127.0.0.1:5000/.

---

## File Structure

project-directory/
|
├── app.py               # Main Flask application
├── templates/
│   └── index.html       # Frontend template for the application
├── static/
│   └── styles.css       # Additional CSS (optional)
├── Procfile             # Deployment file for Heroku
├── README.md            # Project documentation
├── detections.db        # SQLite database (auto-generated)
└── best2.pt             # YOLOv8 model file


---

## Deployment

### Deploy to Heroku
1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).
2. Login to Heroku:
   bash
   heroku login
   
3. Create a Heroku app:
   bash
   heroku create
   
4. Deploy the application:
   bash
   git add .
   git commit -m "Initial commit"
   git push heroku main
   

---

## Usage

1. Open the application in a web browser.
2. View the live webcam feed with object detection.
3. Use the control buttons to play, pause, or stop the feed.
4. Monitor detection records dynamically updated in the table.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

### Contact
For issues, questions, or suggestions, please contact: [your-email@example.com].
