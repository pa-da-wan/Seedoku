import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
from image_utils import *
from digi_model import DigitRecognitionModel
from sudoku_algorithm import SudokuSolver

IMG_HEIGHT = 450
IMG_WIDTH = 450
MODEL_PATH = os.path.join('training', 'digitModel.keras')


def process_frame(frame):
    
    ### Image preprocessing
    processed_img = img_preprocess(frame)

    ### Find contours and get the largest box
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_box = get_largest_box(contours)

    ### Check if a Sudoku-like rectangular box is found
    if largest_box is not None and len(largest_box) == 4:
        ### Calculate the aspect ratio of the bounding box
        x, y, w, h = cv2.boundingRect(largest_box)
        aspect_ratio = w / h

        ### Check if the aspect ratio is close to 1 (indicating a square or nearly square box)
        if 0.9 <= aspect_ratio <= 1.1:
            ### Draw the box on the frame
            cv2.drawContours(frame, [largest_box], -1, (230, 120, 0), 5)

            ### Identify corners
            corners = identify_corners(largest_box)

            ### Perspective transformation
            warped_img = get_perspective_transform(frame, corners, width=IMG_WIDTH, height=IMG_HEIGHT)
            
            ### Split the Sudoku grid into cells
            cells = split_grid(warped_img)

            ### Load the digit recognition model
            digit_recognition_model = DigitRecognitionModel(MODEL_PATH)

            ### Get grid numbers using the model
            puzzle = digit_recognition_model.get_grid_numbers(cells)
            gridNums = puzzle.copy().reshape((1, -1))

            ### Get a mask such that only empty spaces in the puzzle have a mask value of 1
            mask = np.where(gridNums > 0, 0, 1)
            
            ### Create a SudokuSolver instance
            sudoku_solver = SudokuSolver(puzzle)

            ### Solve the Sudoku puzzle
            solution = sudoku_solver.getSudokuSolution()
            ### print('solution: ', np.array(solution).reshape(9, -1))

            ### Get the solution digits
            solNums = np.squeeze(np.array(solution).reshape(1, -1) * mask)
            ### print("solnums: ",solNums)

            ### Create a blank image to add solution numbers onto
            blank_img = np.zeros_like(frame)
            
            
            ### Display the solution on the original image
            solution_img = display_nums(blank_img, solNums)
    
            ### Reverse perspective transformation
            inv_warp_img = get_perspective_transform(solution_img, corners,width =frame.shape[1], height=frame.shape[0], reverse=True)
            
            ### Overlay the solutions onto the Sudoku
            overlaid_img = cv2.addWeighted(frame, 0.6, inv_warp_img, 0.4, -10)

            return overlaid_img
        else:
            ### If the aspect ratio is not close to 1, it's not a Sudoku puzzle
            print(" Sudoku not recognized in the frame.")
            return frame

    # return frame



def main():
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ## Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break
        
        ### Process the frame
        processed_frame = process_frame(frame)

        
        ### Display the frame
        cv2.imshow("Solved puzzle", processed_frame)
        
        ## Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ## Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    