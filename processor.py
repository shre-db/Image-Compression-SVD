import numpy as np
import streamlit as st


class Decompose:
    def __init__(self, img) -> None:
        self.dimensions = len(img.shape)
        self.rank = 0

    def svd_color_img(self, img):
        # Split the image into its RBG channels
        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]

        # Singular Value Decomposition for each channel seperately
        U_red, S_red, Vt_red = np.linalg.svd(red_channel, full_matrices=False)
        U_green, S_green, Vt_green = np.linalg.svd(green_channel, full_matrices=False)
        U_blue, S_blue, Vt_blue = np.linalg.svd(blue_channel, full_matrices=False)

        max_S_red, max_S_green, max_S_blue = S_red.size, S_green.size, S_blue.size
        num_S = min(max_S_red, max_S_green, max_S_blue)
        self.rank = st.slider(
            "**SLIDE TO ADJUST THE RANK**",
            min_value=0, max_value=num_S - 1,
            help="Higher the rank, closer the compressed image is to the original image.")    
        return ((U_red, S_red, Vt_red), (U_green, S_green, Vt_green), (U_blue, S_blue, Vt_blue))
    
    def svd_grayscale_img(self, img):
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(img, full_matrices=False)
        num_S = S.size
        self.rank = st.slider(
            "**SLIDE TO ADJUST THE RANK**",
            min_value=0, max_value=num_S - 1,
            help="Higher the rank, closer the compressed image is to the original image.")
        return U, S, Vt

    def low_rank_approx(self, U, S, Vt):
        """
        A function to compute a lower rank approximation of the image.

        Parameters:
        -----------
        - rank (int): Higher the value of rank, the closer the approximated image is to the original imag
        - U (m x m array): Left Singular Vectors which are the columns of matrix U.
        - S (m x n array): Is a diagonal matrix containing the singular values of A.
        - Vt (n x n array): Right Singular Vectors which are the rows of the matrix Vt.
        """
        S = np.diag(S)
        X_approx = U[:, :self.rank] @ S[:self.rank, :self.rank] @ Vt[:self.rank, :]
        return X_approx
    
    