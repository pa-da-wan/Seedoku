
# Seedoku

Welcome to Seedoku – a Sudoku puzzle solver program with digit recognition!

## Overview

Seedoku is a Python-based Sudoku solver that utilizes digit recognition to solve Sudoku puzzles from videos or a camera.

## Contents
### Structure

- `src/`: Contains the main source code files.
  
  - `digi_model.py`: Contains code that will be used to recognize sudoku board digits.
  - `image_utils.py`: Contains image processing utilities.
  - `seedoku_main.py`: Main application script.
  - `sudoku_algorithm.py`: Sudoku backtracking algorithm.

- `training/`: Includes training-related files and trained digit recognition model.
  - `Digits1/`: Training data for digit recognition.
  - `digitModel.keras`: Trained digit recognition model.
  - `digitPredModel.h5`: Alternate digit recognition model.
  - `digitTraining.ipynb`: Jupyter Notebook for digit recognition training.
  - `logs2/`: Log files from training.
  - `savedModel2/`: Alternate saved model.

- `media/`: Holds sample videos and images.

- `README.md`: You are here.
- `requirements.txt`: Lists project dependencies.



## Demo
1. gif showing step involved
   
![seedokuSteps](https://github.com/pa-da-wan/Seedoku/assets/73534577/35edc08b-74a9-458e-a804-7978a3a8655c "process summary")

2. live demo using laptop webcam
   
![application video](https://github.com/pa-da-wan/Seedoku/assets/73534577/25cf5afc-a220-4613-bb51-6fc990bb8587 "live demo")



## How It Works

Seedoku uses a combination of image processing and a backtracking algorithm to solve Sudoku puzzles. The process involves:

1. **Image Preprocessing**

    Convert to grayscale, apply Gaussian blur, and perform adaptive thresholding.

2. **Detect Sudoku Grid**

    Find contours and identify the largest rectangle with an aspect ratio close to 1.
    
3. **Perspective Transformation**

    Account for the angle-of-view of camera and align the skewed image suitably.

3. **Digit Recognition**

    Split the Sudoku grid into cells.

    Use the trained digit recognition model to predict digits in each cell.
4. **Sudoku Solving**
    
    Implement a backtracking algorithm to solve the Sudoku puzzle.
5. **Overlay Solution**

    Display the solution overlaid on the original image.


## Usage 

Follow the instructions below to set up Seedoku on your local machine.

### Prerequisites

- Python 3.9 or later installed on your machine.

### Installation

- **Clone the Repository:**
   ```bash
    git clone https://github.com/pa-da-wan/Seedoku.git
    ```
    ```bash
    cd Seedoku
    ```
- **Create a Virtual Environment:**

    Choose one of the following methods to create and activate a virtual environment:

    - **for example using Conda:**
      ```bash
      conda create --name seedoku python=3.9
      ```
      ```bash
      conda activate seedoku
      ```

- **Install Required Packages:**

    Install the necessary Python packages by running the following command:

    ```bash
    pip install -r requirements.txt
    ```   

### Run seedoku
  - Open a terminal in the `src/` directory.
  - Execute the main script:
    ```bash
    python seedoku_main.py
    ```


### Digit Recognition Model
The model was trained on over 52000 images corresponding to digits from 0 to 9 for 200 epochs. 

#### Model Performance

|  Metric   |  Accuracy  | Precision   | Recall         |
| --------- | ---------- | ----------- |----------------| 
| **Value** |   0.988    |      0.96   |      0.94      |

## Potential Improvements

1. The current implementation assumes that the largest box in the image, which is approximately square-shaped, is a Sudoku puzzle. A more robust procedure to verify whether the detected box exhibits Sudoku features could be implemened which may involve analyzing grid lines, checking for a specific number of cells, and ensuring proper alignment.
2. Enhance the program's accuracy on low-quality video streams by implementing strategies such as noise reduction, image enhancement, and advanced preprocessing techniques.
3. Addressing occasional lag during frame processing is crucial for real-time applications. Evaluate the current code for potential bottlenecks, consider optimizing image preprocessing steps, and explore parallelization or concurrency techniques to speed up the overall processing pipeline.
