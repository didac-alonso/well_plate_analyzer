# Image analyzer for well plate experiment results

# Installation Guide

To set up and run this project, follow the steps below.

## Prerequisites
Make sure you have **Python 3.7+** installed on your system. You can check your Python version by running:

```bash
python --version
```

## 1. Clone the Repository
First, clone this repository to your local machine:

```bash
git clone <repository_url>
cd <repository_folder>
```

## 2. Create a virtual environment
This would protect you libraries from conflicts
```sh
python -m venv venv
```

## 3. Activate the virtual environment
```sh
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

## 4. Install dependencies
```sh
pip install -r requirements.txt
```

## 5. Run the program
```sh
python image_analyzer.py
```

# How to use?
## 1. Activate the virtual enviroment
```sh
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

## 2. Run the program
```sh
python image_analyzer.py
```

## 3. Select the image to analyze:
A explorer/finder window will open asking to select the image to analyze.

![alt text](image.png)

## 4. Cut and ajust the perspective of the image:
The image will be preprocessed to eliminate some reflexes, and it will be shown in your screen. You must click in every corner, like showing where the limits of the well plate are. Just like in the example bellow:

![alt text](image-1.png)

You have to do this for all the corners. Then the program will automatically rotate and change the perspective to get the image as regular and straight as possible.

## 5. Select the first 3 wells centres: Top left, the one next to it and the one under it.
This is used to interpolate the position of the rest of all well centres, be precise.

![alt text](image-3.png)

## 6. Insert in the terminal how many horizontal and vertical wells do you have in your matrix.

![alt text](image-2.png)

## 7. The results are saved in well_size.csv and well_white_intensity.csv, following the same structure as the well matrix.

![alt text](<Screenshot 2025-02-13 at 16.42.56.png>)