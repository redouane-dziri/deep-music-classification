import numpy as np

from skimage.feature.texture import greycomatrix


def generate_glcm(array, distance, angle_in_deg):
    """This function computes a grey level co-occurrence map from a quantized
    map for a given list of angles and a distance
    
    Arguments:
        array {np.array} -- an int array quantized map
        distance {int} -- the distance offset for computation of co-occurrence
        angle_in_deg {int} -- integer degree angle between pairs to be 
            considered for co-occurrence
    
    Returns:
        glcm {np.array} -- a uint32 array in which the value 
            glcm[i, j] is the number of times that grey-level j occurs at a 
            distance `d` and at an angle `theta` from grey-level i. 
    """
    glcm = greycomatrix(
        array, 
        distances=[distance], 
        angles=[angle_in_deg * (np.pi/180)],
        symmetric=False, 
        normed=False
    )
    
    # remove two superfluous dimensions from `glcm`
    return np.squeeze(glcm)
