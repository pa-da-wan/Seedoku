import cv2
import numpy as np

def img_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    invert = cv2.bitwise_not(thresh)
    kernel = np.ones((1, 1))
    opening = cv2.morphologyEx(invert, cv2.MORPH_OPEN, kernel)
    return opening

def get_largest_box(contours):
    largest_box = None
    
    # Iterate through contours
    for cnt in contours:
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
    
        # Check if the polygon has four vertices (a rectangle)
        if len(approx) == 4:
            # Find the largest rectangle
            if largest_box is None or cv2.contourArea(approx) > cv2.contourArea(largest_box):
                # if cv2.contourArea(approx)>10000:

                    largest_box = approx

    return largest_box

def identify_corners(largest_box):
    corners = np.zeros_like(np.squeeze(largest_box))
    ##Steps:
    ## Calculate the centroid of the rectangle
    ##Calculate the angles of the lines connecting each vertex to the centroid
    ## Sort vertices based on angles
    ## Arrange corners in the correct order
    vertices = np.squeeze(largest_box)
    
    centroid = np.mean(vertices, axis=0)
    
    
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    
    
    sorted_indices = np.argsort(angles)
    
    top_left = vertices[sorted_indices[0]]
    top_right = vertices[sorted_indices[1]]
    bottom_right = vertices[sorted_indices[2]]
    bottom_left = vertices[sorted_indices[3]]
    corners[0] = top_left
    corners[1] = top_right
    corners[2] = bottom_left
    corners[3] = bottom_right
    return np.float32(corners)


def get_perspective_transform(image, source_pts, width=450, height=450, reverse=False):
    destination = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    
    if reverse:
        source_pts, destination = destination, source_pts
    
    xform_matrix = cv2.getPerspectiveTransform(np.float32(source_pts), np.float32(destination))
    xformed_img = cv2.warpPerspective(image, xform_matrix, (width, height))
    
    return xformed_img

def split_grid(image):
    cells =[]
    rows = np.vsplit(image,9)
    for r in rows:
        cols= np.hsplit(r,9)
        for cell in cols:
            cells.append(cell)
    return cells
    


def display_nums(image, number_list, text_color=(10, 255, 0)):
    """
    Display numbers on a 9x9 grid on the given image.

    Parameters:
    - image: numpy array representing the image
    - number_list: a list of 81 numbers representing a 9x9 grid
    - text_color: tuple representing the RGB color for the displayed text

    Returns:
    - image with numbers displayed
    """
    # Calculate the width and height of each section in the grid
    section_width = int(image.shape[1] / 9)
    section_height = int(image.shape[0] / 9)

    # Iterate through each cell in the 9x9 grid
    for row in range(9):
        for col in range(9):
            # Calculate the index in the flattened list representation of the grid
            index = (col * 9) + row

            # Check if the number is not zero (indicating a filled cell)
            if number_list[index] != 0:
                # Calculate the position to display the text in the center of the cell
                text_position = (row * section_width + int(section_width / 2) - 10,
                                 int((col + 0.8) * section_height))

                # Draw the text on the image
                cv2.putText(image, str(number_list[index]),
                            text_position, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            5, text_color, 2, cv2.LINE_AA)

    # Return the modified image with numbers displayed
    return image

def displaySuccessMsg(image, message="Sudoku Solved!!", text_color=(255, 255, 255), font_size=5):
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Choose the font type and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = font_size

    # Get the size of the text
    text_size = cv2.getTextSize(message, font, scale, 2)[0]

    # Calculate the position to display the text in the center
    text_position = ((width - text_size[0]) // 2, (height + text_size[1]) // 2)

    # Draw the text on the image
    cv2.putText(image, message, text_position, font, scale, text_color, 5, cv2.LINE_AA)

    return image