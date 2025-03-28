# import webrtc
import streamlit as st

# from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import numpy as np
import av

st.title("OpenCV Filters on Video Stream")

# filter = "none"


# def transform(frame: av.VideoFrame):
#     img = frame.to_ndarray(format="bgr24")

#     if filter == "blur":
#         img = webrtc.GaussianBlur(img, (21, 21), 0)
#     elif filter == "canny":
#         img = webrtc.cvtColor(webrtc.Canny(img, 100, 200), webrtc.COLOR_GRAY2BGR)
#     elif filter == "grayscale":
#         # We convert the image twice because the first conversion returns a 2D array.
#         # the second conversion turns it back to a 3D array.
#         img = webrtc.cvtColor(webrtc.cvtColor(img, webrtc.COLOR_BGR2GRAY), webrtc.COLOR_GRAY2BGR)
#     elif filter == "sepia":
#         kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
#         img = webrtc.transform(img, kernel)
#     elif filter == "invert":
#         img = webrtc.bitwise_not(img)
#     elif filter == "none":
#         pass

#     return av.VideoFrame.from_ndarray(img, format="bgr24")


# col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

# with col1:
#     if st.button("None"):
#         filter = "none"
# with col2:
#     if st.button("Blur"):
#         filter = "blur"
# with col3:
#     if st.button("Grayscale"):
#         filter = "grayscale"
# with col4:
#     if st.button("Sepia"):
#         filter = "sepia"
# with col5:
#     if st.button("Canny"):
#         filter = "canny"
# with col6:
#     if st.button("Invert"):
#         filter = "invert"


# webrtc_streamer(key="streamer", video_frame_callback=transform, sendback_audio=False)


from streamlit_webrtc import webrtc_streamer

webrtc_streamer(key="streamer", sendback_audio=False, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})  # Add this config
