import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
st.set_page_config(layout='wide', page_title="ImageSVD", page_icon="icons/angle-down-solid.svg", initial_sidebar_state='collapsed')
import time
from processor import Decompose

st.markdown('<h1 style="text-align: center;"><i class="fa-solid fa-angle-down"></i> &nbspImageSVD </h1>', unsafe_allow_html=True)
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
image = st.file_uploader("**SELECT AN IMAGE**", type=['jpg'], help="Images > 1MB require considerable time for compression with our current resources. Please hang tight!")
st.markdown('')
st.markdown('')

if image is not None:
    pil_img = Image.open(image)
    img = np.array(pil_img)
    
    decomposer = Decompose(img=img)

    if decomposer.dimensions == 2:
        U, S, Vt = decomposer.svd_grayscale_img(img)

        compressed_image = decomposer.low_rank_approx(U=U, S=S, Vt=Vt).astype(np.uint8)

        compressed_image = Image.fromarray(compressed_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Original image", use_column_width=True)
        with col2:
            st.image(compressed_image, caption="Compressed Image", use_column_width=True)

    elif decomposer.dimensions == 3:
        r, g, b = decomposer.svd_color_img(img)

        compressed_red = decomposer.low_rank_approx(U=r[0], S=r[1], Vt=r[2])
        compressed_green = decomposer.low_rank_approx(U=g[0], S=g[1], Vt=g[2])
        compressed_blue = decomposer.low_rank_approx(U=b[0], S=b[1], Vt=b[2])

        compressed_image = np.clip(np.stack((compressed_red, compressed_green, compressed_blue), axis=-1), 0, 255).astype(np.uint8)

        compressed_image = Image.fromarray(compressed_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Original image", use_column_width=True)
        with col2:
            st.image(compressed_image, caption="Compressed Image", use_column_width=True)
    
    else:
        st.warning("Image not compatible with our system!", icon='⚠')

    
    st.markdown('***')
    st.markdown('')

    compressed_image.save("images/Compressed_Image.jpg", "JPEG")
    with open("images/Compressed_Image.jpg", "rb") as file:
        btn = st.download_button(
            label="**Download Compressed Image**",
            data=file,
            file_name="Compressed_Image.jpg",
            mime="image/jpg"
        )

    st.markdown('')
    if st.toggle("**Advanced Info**"):
        if decomposer.dimensions == 2:
            st.markdown('')
            st.markdown('')
            st.markdown('### Singular Value Analysis')
            st.markdown('')
            with st.spinner("Plotting..."):
                time.sleep(2)
            fig1, ax = plt.subplots(1, 1, figsize=(12, 4))

            ax.semilogy(S, color='orange')
            ax.set_title("Singular Values")
            ax.set_xlabel('$j$')
            ax.set_ylabel('$log\sigma_j$')
            ax.grid(alpha=0.2)

            st.pyplot(fig=fig1)
            
            st.markdown('')
            st.markdown("***Singular Values $\sigma_j$ in logarithmic scale vs. index of the singular values $j$***")
            st.markdown("The above plots are displayed using a logarithmic scale on the y-axis (`semilogy()` function), which is useful for visualizing a wide range of values. Singular values represent the importance or weight of each singular vector in the decomposition. Singular values are typically ordered in descending order, so the first few singular values capture most of the data's variance, and the rest contribute less. This type of plot helps you see the relative differences between singular values more clearly, especially when there are significant variations in magnitude.")
            st.markdown('')
            st.markdown('***')

            fig2, ax = plt.subplots(1, 1, figsize=(12, 4))

            ax.plot(np.cumsum(S/np.sum(S)), color='orange')
            ax.set_title('Singular Values: Cumulative Sum')
            ax.set_xlabel('$j$')
            ax.set_ylabel('Cumulative Sum')
            ax.vlines(x=decomposer.rank, ymin=0, ymax=np.cumsum(S/np.sum(S))[decomposer.rank], color='magenta', linestyle='--')
            ax.hlines(y=np.cumsum(S/np.sum(S))[decomposer.rank], xmin=0.0, xmax=decomposer.rank, color='magenta', linestyle='--')
            ax.plot(decomposer.rank, np.cumsum(S/np.sum(S))[decomposer.rank], 'mo')
            ax.text(
                x=decomposer.rank + 10,
                y=np.cumsum(S/np.sum(S))[decomposer.rank] - 0.05,
                s=f"{round(np.cumsum(S/np.sum(S))[decomposer.rank]*100, 1)} %"
            )
            plt.grid(alpha=0.2)

            st.pyplot(fig=fig2)

            st.markdown('')
            st.markdown("***Cumulative Sum (of squared Singular values per total sum of squared Singular Values) vs. index of the singular values $j$***")
            st.markdown("This plot helps you understand the contribution of each singular value to the total variance or energy of the matrix. The cumulative sum curve shows how much of the total variance is captured by including the first $j$ singular values. If you were to draw a horizontal line at a certain value on the y-axis, the corresponding x-axis value would tell you how many singular values you need to include to capture that percentage of the total variance and vice versa.")


        elif decomposer.dimensions == 3:
            st.markdown('')
            st.markdown('')
            st.markdown('### Singular Value Analysis')
            st.markdown('')
            with st.spinner("Plotting..."):
                time.sleep(2)
            fig1, ax = plt.subplots(1, 3, figsize=(18, 6))

            ax[0].semilogy(r[1], color='red')
            ax[0].set_title("Singular Values: Red Channel")
            ax[0].set_xlabel('$j$')
            ax[0].set_ylabel('$log\sigma_j$')
            ax[0].grid(alpha=0.2)

            ax[1].semilogy(g[1], color='green')
            ax[1].set_title("Singular Values: Green Channel")
            ax[1].set_xlabel('$j$')
            ax[1].set_ylabel('$log\sigma_j$')
            ax[1].grid(alpha=0.2)

            ax[2].semilogy(b[1], color='blue')
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

            ax[0].plot(np.cumsum(r[1]/np.sum(r[1])), color='red')
            ax[0].set_title('Singular Values [Red Channel]: Cumulative Sum')
            ax[0].set_xlabel('$j$')
            ax[0].set_ylabel('Cumulative Sum')
            ax[0].vlines(x=decomposer.rank, ymin=0, ymax=np.cumsum(r[1]/np.sum(r[1]))[decomposer.rank], color='magenta', linestyle='--')
            ax[0].hlines(y=np.cumsum(r[1]/np.sum(r[1]))[decomposer.rank], xmin=0.0, xmax=decomposer.rank, color='magenta', linestyle='--')
            ax[0].plot(decomposer.rank, np.cumsum(r[1]/np.sum(r[1]))[decomposer.rank], 'mo')
            ax[0].text(
                x=decomposer.rank + 10,
                y=np.cumsum(r[1]/np.sum(r[1]))[decomposer.rank] - 0.05,
                s=f"{round(np.cumsum(r[1]/np.sum(r[1]))[decomposer.rank]*100, 1)} %"
            )
            ax[0].grid(alpha=0.2)

            ax[1].plot(np.cumsum(g[1]/np.sum(g[1])), color='green')
            ax[1].set_title('Singular Values [Green Channel]: Cumulative Sum')
            ax[1].set_xlabel('$j$')
            ax[1].set_ylabel('Cumulative Sum')
            ax[1].vlines(x=decomposer.rank, ymin=0, ymax=np.cumsum(g[1]/np.sum(g[1]))[decomposer.rank], color='magenta', linestyle='--')
            ax[1].hlines(y=np.cumsum(g[1]/np.sum(g[1]))[decomposer.rank], xmin=0.0, xmax=decomposer.rank, color='magenta', linestyle='--')
            ax[1].plot(decomposer.rank, np.cumsum(g[1]/np.sum(g[1]))[decomposer.rank], 'mo')
            ax[1].text(
                x=decomposer.rank + 10,
                y=np.cumsum(g[1]/np.sum(g[1]))[decomposer.rank] - 0.05,
                s=f"{round(np.cumsum(g[1]/np.sum(g[1]))[decomposer.rank]*100, 1)} %"
            )
            ax[1].grid(alpha=0.2)

            ax[2].plot(np.cumsum(b[1]/np.sum(b[1])), color='blue')
            ax[2].set_title('Singular Values [Blue Channel]: Cumulative Sum')
            ax[2].set_xlabel('$j$')
            ax[2].set_ylabel('Cumulative Sum')
            ax[2].vlines(x=decomposer.rank, ymin=0, ymax=np.cumsum(b[1]/np.sum(b[1]))[decomposer.rank], color='magenta', linestyle='--')
            ax[2].hlines(y=np.cumsum(b[1]/np.sum(b[1]))[decomposer.rank], xmin=0.0, xmax=decomposer.rank, color='magenta', linestyle='--')
            ax[2].plot(decomposer.rank, np.cumsum(b[1]/np.sum(b[1]))[decomposer.rank], 'mo')
            ax[2].text(
                x=decomposer.rank + 10,
                y=np.cumsum(b[1]/np.sum(b[1]))[decomposer.rank] - 0.05,
                s=f"{round(np.cumsum(b[1]/np.sum(b[1]))[decomposer.rank]*100, 1)} %"
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

css_copyr = '''
<style>
.footer-text {
    font-size: 15px;
    color: #888888;
    text-align: center;
}
</style>
'''

st.markdown(css_copyr, unsafe_allow_html=True)

st.markdown('<p class="footer-text">Copyright © 2023 &nbsp<i class="fa-solid fa-angle-down"></i>&nbspImageSVD</p>', unsafe_allow_html=True)
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
</div><br>
<div>
    <p class="footer-text">Version 1.0.0</p>
</div>
'''

st.markdown(css_fa, unsafe_allow_html=True)