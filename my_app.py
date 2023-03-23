import streamlit as st 
import os 
from PIL import Image
from obj_det_and_trk_streamlit import detect
from ob_detect import run
import time 
st.sidebar.image("2.png")

with st.sidebar : 
    select_type_detect = st.selectbox("Detection and Tracking from :  ",
                                            ("File", "Live"))
    if select_type_detect == "Live" :          
        model_path = "afd_dataset_big_smokes.pt"
    else : 
        model_path = "WILDFIRESMOKEBEST.pt"
    select_device = st.selectbox("Select compute Device :  ",
                                            ("CPU", "GPU"))
    save_output_video = st.radio("Save output video?",
                                            ('Yes', 'No'))
    if save_output_video == 'Yes':
        nosave = False
        display_labels=True
    else:
        nosave = True
        display_labels = True
    conf_thres = st.text_input("Class confidence threshold", "0.25")

if select_device == "GPU" : 
    DEVICE_NAME = st.selectbox("Select CUDA index : " , 
                                     (0, 1 , 2)) 
elif select_device =="CPU" : 
    DEVICE_NAME = "cpu"


def fromvid(source, conf_thres , device , vid_id , nosave  , display_labels) :
    kpi5, kpi6 = st.columns(2)
    with kpi5:
        st.markdown("""<h5 style="color:white;">
                                  CPU Utilization</h5>""", 
                                  unsafe_allow_html=True)
        kpi5_text = st.markdown("0")
    with kpi6:
        st.markdown("""<h5 style="color:white;">
                                Memory Usage</h5>""", 
                                unsafe_allow_html=True)
        kpi6_text = st.markdown("0")
    stframe = st.empty()
    detect(weights=model_path,
                   source=source,
                   stframe=stframe, 
                   kpi5_text=kpi5_text , 
                   kpi6_text=kpi6_text, 
                   conf_thres=float(conf_thres),
                   device=device,
                   hide_labels=False,  
                   hide_conf=False,
                   project=vid_id, 
                   nosave=nosave, 
                   display_labels=display_labels)

if select_type_detect == "File" :
    tab0 , tab1, tab2 = st.tabs([  "Home", 
                                   "Image",
                                   "Video"])
    with tab0 : 
        st.header("About OUR Project : ")
        st.image("smokey.gif")
        st.write("""The Wildfire Smoke Detection and Tracking System is a high-tech solution that uses drones or HPWREN cameras equipped with the YOLOv5 algorithm to detect and alert individuals of potential wildfire smoke in the surrounding area. 
        The system is designed to work in real-time, constantly monitoring the environment and providing accurate and timely notifications of any smoke detected. The use of drones allows for a wider coverage area and the ability to access hard-to-reach areas while HPWREN cameras are utilized to monitor remote areas that are at high risk of wildfire.
        The YOLOv5 algorithm is a powerful machine learning tool that is able to detect and classify objects in real-time, making it a valuable tool in detecting smoke and preventing the spread of wildfire.""")
        st.header("About Dataset : ")
        st.write("This dataset is released by AI for Mankind in collaboration with HPWREN under a Creative Commons by Attribution Non-Commercial Share Alike license. The original dataset (and additional images without bounding boxes) can be found in their GitHub repo.")
        st.write("https://github.com/aiformankind/wildfire-smoke-dataset")
    with tab1:
        image_id = str(time.asctime())
        st.header("Image")
        image_upload = st.file_uploader("upload your image" , 
                                        type=["png" , 
                                               "jpg"])
        try : 
            run(weights=model_path, 
                source=image_upload.name,
                device=DEVICE_NAME ,
                name=image_id)
            image = Image.open(f"{image_id}/{image_upload.name}")
            st.image(image)
        except : 
            pass
    
    with tab2 :
        vid_id = str(time.asctime())
        st.header("Video")
        video_upload = st.file_uploader("upload your Video" ,
                                         type=['mp4', 'mov'])
        if video_upload : 
            if st.button('Click to Start'):
                fromvid(source=video_upload.name ,  
                        vid_id=vid_id , 
                        device=DEVICE_NAME ,
                        conf_thres=conf_thres ,
                        nosave=nosave ,
                        display_labels=display_labels)
                
                

elif select_type_detect == "Live" : 
    live_id = str(time.asctime())
    live_type = st.selectbox("Live type :  ",
                             ("URL", "LOCAL-CAM"))
    if live_type=="URL":
        url = st.text_input('Entre your URL Stream : ')
        if url : 
            fromvid(source=url , 
                    vid_id=live_id ,
                    device=DEVICE_NAME , 
                    conf_thres=conf_thres ,
                    nosave=nosave , 
                    display_labels=display_labels )
    elif live_type=="LOCAL-CAM" : 
        cam_id = str(time.asctime())
        index = st.selectbox("Select device index : " , 
                                        (0, 1 , 2))
        if st.button('Run Detection'):           
            os.system(f"python obj_det_and_trk.py --source {index} --view-img --conf-thres {conf_thres} --device {DEVICE_NAME} --project '{cam_id}' --color-box")
