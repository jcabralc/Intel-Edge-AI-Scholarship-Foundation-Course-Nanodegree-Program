import cv2
import numpy as np

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)
    # TODO: Preprocess the image for the pose estimation model
    W = 456
    H = 256
    resized = cv2.resize(preprocessed_image, (W, H))
    resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    preprocessed_image = resized.reshape((1, 3, H, W))
    
    return preprocessed_image


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)
    # TODO: Preprocess the image for the text detection model
    W = 1280
    H = 768
    resized = cv2.resize(preprocessed_image, (W, H))
    resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    preprocessed_image = resized.reshape((1, 3, H, W))
    
    return preprocessed_image


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the car metadata model
    W = 72
    H = 72
    resized = cv2.resize(preprocessed_image, (W, H))
    resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    preprocessed_image = resized.reshape((1, 3, H, W))
    return preprocessed_image
