# Wall Crack Detection

This project uses a deep learning model to detect cracks in walls using real-time video feed from a mobile phone camera. 
The project is built using Tensorflow, Keras, and OpenCV.


## Project Structure
```
|-- crack_detector.h5 # Trained Model
|--main.py # Real-time detection script
|--concrete_data/ #Dataset folder (if training from scratch is required)
||--train/
|||-Positive/
|||-Negative/
||--val/
|||-Positive/
|||-Negative/
|--detection_log.csv # Optional, it will help you to store predictions
|--README.md
|-requirements.txt
```
Requirements

```
- python 3.10+
- Tensorflow 2.x
- OpenCV
- Numpy
```
Install Dependencies using pip:
```
pip install -r requirements.txt
```
For better experience make a virtual environment.
