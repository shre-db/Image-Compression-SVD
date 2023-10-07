import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from PIL import Image
import streamlit as st
st.set_page_config(layout='wide')
import time

st.markdown("<h1 style='text-align: center;'>Introducing ImageSVD</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'><i>Your Image Compression Solution</i></h2>", unsafe_allow_html=True)
st.markdown('')
st.markdown("""


Welcome to ImageSVD, your go-to web application for efficient image compression using Singular Value Decomposition (SVD). In a world where visual content is key, we understand the importance of striking a balance between image quality and file size. With ImageSVD, you can easily achieve just that.

Our user-friendly platform empowers you to reduce the size of your images while preserving their visual integrity. Leveraging the power of SVD, ImageSVD breaks down your images into their essential components, allowing you to control the level of compression. Whether you're a photographer, web developer, or simply someone who values optimized visuals, ImageSVD is here to simplify your image compression journey.

Key Features:

- **Customizable Compression:** Choose the level of compression that suits your needs.
- **Preserve Image Quality:** Reduce file sizes without compromising on image clarity.
- **User-Friendly Interface:** A simple and intuitive platform for users of all levels.
- **Multi-Channel Support:** Compress both grayscale and color images with ease.
            
Discover the perfect balance between image quality and file size with ImageSVD. Start compressing your images today and unlock a world of efficient visual content.
            """)
st.markdown('')
st.markdown('')
image = st.file_uploader("**SELECT AN IMAGE**", type=['jpg'])
st.markdown('')
st.markdown('')
st.markdown('')

if image is not None:
    pil_img = Image.open(image)
    img = np.array(pil_img)

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
    rank = st.slider("**SLIDE TO ADJUST THE RANK**", min_value=0, max_value=num_S, help="Higher the rank, closer the compressed image is to the original image.")


    def low_rank_approx(rank, U, S, Vt):
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
        X_approx = U[:, :rank] @ S[:rank, :rank] @ Vt[:rank, :]
        return X_approx

    
    compressed_red = low_rank_approx(rank, U_red, S_red, Vt_red)
    compressed_green = low_rank_approx(rank, U_green, S_green, Vt_green)
    compressed_blue = low_rank_approx(rank, U_blue, S_blue, Vt_blue)

    compressed_image = np.clip(np.stack((compressed_red, compressed_green, compressed_blue), axis=-1), 0, 255).astype(np.uint8)

    compressed_image = Image.fromarray(compressed_image)
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original image", use_column_width=True)
    with col2:
        st.image(compressed_image, caption="Compressed Image", use_column_width=True)

    st.markdown('***')
    st.markdown('')
    if st.button("**Download Compressed Image**"):
        compressed_image.save("images/Compressed_Image.jpg", "JPEG")

    # if st.download_button(
    #     "**Download Compressed Image**",
    #     data=compressed_image,
    #     file_name="Compressed-Image.jpg",
    #     mime="JPG/JPEG",
    # ):
    #     pass
    st.markdown('')
    if st.toggle("**Advanced Info**"):
        st.markdown('')
        st.markdown('')
        st.markdown('### Singular Value Analysis')
        st.markdown('')
        with st.spinner("Plotting..."):
            time.sleep(2)
        fig1, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].semilogy(S_red, color='red')
        ax[0].set_title("Singular Values: Red Channel")
        ax[0].set_xlabel('$j$')
        ax[0].set_ylabel('$log\sigma_j$')
        ax[0].grid(alpha=0.2)

        ax[1].semilogy(S_green, color='green')
        ax[1].set_title("Singular Values: Green Channel")
        ax[1].set_xlabel('$j$')
        ax[1].set_ylabel('$log\sigma_j$')
        ax[1].grid(alpha=0.2)

        ax[2].semilogy(S_blue, color='blue')
        ax[2].set_title("Singular Values: Blue Channel")
        ax[2].set_xlabel('$j$')
        ax[2].set_ylabel('$log\sigma_j$')
        ax[2].grid(alpha=0.2)


        st.pyplot(fig=fig1)

        st.markdown('')
        st.markdown("***Singular Values $\sigma_j$ in logarithmic scale vs. index of the singular values $j$***")
        st.markdown("The above plots are displayed using a logarithmic scale on the y-axis (`semilogy()` function), which is useful for visualizing a wide range of values. Singular values represent the importance or weight of each singular vector in the decomposition. Singular values are typically ordered in descending order, so the first few singular values capture most of the data's variance, and the rest contribute less. This type of plot helps you see the relative differences between singular values more clearly, especially when there are significant variations in magnitude.")
        st.markdown('')
        st.markdown('***')

        fig2, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].plot(np.cumsum(S_red/np.sum(S_red)), color='red')
        ax[0].set_title('Singular Values [Red Channel]: Cumulative Sum')
        ax[0].set_xlabel('$j$')
        ax[0].set_ylabel('Cumulative Sum')
        ax[0].vlines(x=rank, ymin=0, ymax=np.cumsum(S_red/np.sum(S_red))[rank], color='magenta', linestyle='--')
        ax[0].hlines(y=np.cumsum(S_red/np.sum(S_red))[rank], xmin=0.0, xmax=rank, color='magenta', linestyle='--')
        ax[0].plot(rank, np.cumsum(S_red/np.sum(S_red))[rank], 'wo')
        ax[0].text(
            x=rank + 10,
            y=np.cumsum(S_red/np.sum(S_red))[rank] - 0.05,
            s=f"{round(np.cumsum(S_red/np.sum(S_red))[rank]*100, 1)} %"
        )
        ax[0].grid(alpha=0.2)

        ax[1].plot(np.cumsum(S_green/np.sum(S_green)), color='green')
        ax[1].set_title('Singular Values [Green Channel]: Cumulative Sum')
        ax[1].set_xlabel('$j$')
        ax[1].set_ylabel('Cumulative Sum')
        ax[1].vlines(x=rank, ymin=0, ymax=np.cumsum(S_green/np.sum(S_green))[rank], color='magenta', linestyle='--')
        ax[1].hlines(y=np.cumsum(S_green/np.sum(S_green))[rank], xmin=0.0, xmax=rank, color='magenta', linestyle='--')
        ax[1].plot(rank, np.cumsum(S_green/np.sum(S_green))[rank], 'wo')
        ax[1].text(
            x=rank + 10,
            y=np.cumsum(S_green/np.sum(S_green))[rank] - 0.05,
            s=f"{round(np.cumsum(S_green/np.sum(S_green))[rank]*100, 1)} %"
        )
        ax[1].grid(alpha=0.2)

        ax[2].plot(np.cumsum(S_blue/np.sum(S_blue)), color='blue')
        ax[2].set_title('Singular Values [Blue Channel]: Cumulative Sum')
        ax[2].set_xlabel('$j$')
        ax[2].set_ylabel('Cumulative Sum')
        ax[2].vlines(x=rank, ymin=0, ymax=np.cumsum(S_blue/np.sum(S_blue))[rank], color='magenta', linestyle='--')
        ax[2].hlines(y=np.cumsum(S_blue/np.sum(S_blue))[rank], xmin=0.0, xmax=rank, color='magenta', linestyle='--')
        ax[2].plot(rank, np.cumsum(S_blue/np.sum(S_blue))[rank], 'wo')
        ax[2].text(
            x=rank + 10,
            y=np.cumsum(S_blue/np.sum(S_blue))[rank] - 0.05,
            s=f"{round(np.cumsum(S_blue/np.sum(S_blue))[rank]*100, 1)} %"
        )
        ax[2].grid(alpha=0.2)
        st.pyplot(fig=fig2)

        st.markdown('')
        st.markdown("***Cumulative Sum (of squared Singular values per total sum of squared Singular Values) vs. index of the singular values $j$***")
        st.markdown("This plot helps you understand the contribution of each singular value to the total variance or energy of the matrix. The cumulative sum curve shows how much of the total variance is captured by including the first $j$ singular values. If you were to draw a horizontal line at a certain value on the y-axis, the corresponding x-axis value would tell you how many singular values you need to include to capture that percentage of the total variance and vice versa.")
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('***')
# st.write("© 2023 Shreyas. All Rights Reserved.")
# st.write("Contact us at shreyasdb99@gmail.com")
st.markdown(
"""
<style>
.footer-text {
    font-size: 15px;
    color: #888888;
    text-align: center;
}
</style>
""",
unsafe_allow_html=True,
)

st.markdown("<p class='footer-text'>© 2023 Shreyas. All Rights Reserved.</p>", unsafe_allow_html=True)
st.markdown("<p class='footer-text'>Contact us at shreyasdb99@gmail.com</p>", unsafe_allow_html=True)
st.markdown('')
css_fa = '''                                                                                                                                                     
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
.footer-fa {
    font-size: 20px;
    color: #888888;
    text-align: center;
    margin: 0 5px;
    display: inline-block;
}
.footer-icons {
    text-align: center;
}
</style>
<div class="footer-icons">                                                                                                                                                                                                                                                                                               
    <a href="https://github.com/shre-db" target="_blank"><i class="fa-brands fa-github footer-fa"></i></a>                                                                                                                                                                
    <a href="https://www.linkedin.com/in/shreyas-bangera-aa8012271/" target="_blank"><i class="fa-brands fa-linkedin footer-fa"></i></a>                                                                                                                                                                         
    <a href="https://www.instagram.com/shryzium/" target="_blank"><i class="fa-brands fa-instagram footer-fa"></i></a>
</div>
'''

st.markdown(css_fa, unsafe_allow_html=True)