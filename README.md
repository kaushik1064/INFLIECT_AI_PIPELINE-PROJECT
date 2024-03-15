# Yolov5 object detection model deployment using flask


### Steps to run Code

- Create a Virtual Environment before running the code(make sure python version is greater than 3.7)
```
conda create -p venv python==3.10 -y
```
- Activate the virtual environment
```
conda activate venv/
```

- Clone the repository.
```
git clone https://github.com/kaushik1064/INFLIECT_AI_PIPELINE-PROJECT.git
```
- Goto the cloned folder.
```
cd INFLIECT_AI_PIPELINE-PROJECT

```
- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Run the code with mentioned command below.



## Web app
Simple app consisting of a form where you can upload an image, and see the inference result of the model in the browser. Run:

` python3 webapp.py --port 5000`

then visit http://localhost:5000/ in your browser:

<p align="center">
<img src="https://github.com/noorkhokhar99/yolov5-flask-object-detection/blob/main/static/yolo_prtscr.png" width="450">
</p>

<p align="center">
<img src="https://github.com/noorkhokhar99/yolov5-flask-object-detection/blob/main/static/uploaded_image1_result.jpg" width="450">
</p>





