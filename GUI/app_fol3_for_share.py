import streamlit as st
from streamlit_folium import folium_static
import folium
from folium import  FeatureGroup
import pandas as pd
from branca.element import Figure


from PIL import Image
import numpy as np 
import streamlit as st 
import pandas as pd
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf



import torch
import pandas as pd
import shutil
import io
import numpy as np
import ast
import cv2
import os
from tqdm.auto import tqdm
import shutil as sh
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

#from common import DetectMultiBackend
from models.common import DetectMultiBackend

from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import detect as det
from types import SimpleNamespace




# Create the title for the web app
st.title("Система видеомониторинга и прогнозирования событий")

st.subheader("Карта с камерами")

st.write("""На приведенной ниже карте представлены все камеры видеонаблюдения. 
    Зеленым отмечены камеры, по которым не было обнаружено событий, связанных с превышением мусора,
    а красным - камеры, по которым были установлены превышения мусора в мусорных баках""")

data = pd.read_csv("geo_cam_done.csv")


fig2=Figure(width=700,height=450)
m1 = folium.Map(location=[55.796134, 49.106405], zoom_start=7)
fig2.add_child(m1)
folium.TileLayer('Stamen Terrain').add_to(m1)
folium.TileLayer('Stamen Toner').add_to(m1)
folium.TileLayer('Stamen Water Color').add_to(m1)
folium.TileLayer('cartodbpositron').add_to(m1)
folium.TileLayer('cartodbdark_matter').add_to(m1)

# add marker for Liberty Bell
l = ["red", "red", "green", "red", "green", "green", "green", "green", "green", "green", 
"red", "green", "red", "green", "green", "green", "green" ]

data2 = data[data["Type"] == "ТКО"]




feature_group1 = FeatureGroup(name='Камеры без превышений')
feature_group2 = FeatureGroup(name='Камеры с обнаруженными превышениями')


for i, val in enumerate(l):
    if val == "green":
        tooltip = data["Адрес установки камеры"][i]
        folium.Marker( [data.iloc[i,-2], data.iloc[i,-1]], 
        popup=data["Адрес установки камеры"][i], tooltip=tooltip, 
        icon=folium.Icon(color=l[i],icon='ok-sign')).add_to(feature_group1)
    else:
        tooltip = data["Адрес установки камеры"][i]
        folium.Marker( [data.iloc[i,-2], data.iloc[i,-1]], 
        popup=data["Адрес установки камеры"][i], tooltip=tooltip, 
        icon=folium.Icon(color=l[i],icon='info-sign')).add_to(feature_group2)
        folium.Circle(radius=200,
        location= [data.iloc[i,-2], data.iloc[i,-1]], 
        popup='Выделенная территория',
        color='red', fill=True).add_to(m1)

    
m1.add_child(feature_group1)
m1.add_child(feature_group2)
m1.add_child(folium.map.LayerControl())

folium_static(m1)








st.subheader("Выбор камер, по которым обнаружены превышения в мусорных контейнерах")
#uploaded_files = st.file_uploader("Нажмите на кнопку, чтобы выбрать камеры, по которым обнаружены превышения в мусорных контейнерах и загрузить изображения с них", 
#       accept_multiple_files=True)
if st.button("Нажмите на кнопку, чтобы выбрать камеры, по которым обнаружены превышения в мусорных контейнерах и загрузить изображения с них"):
    files = os.listdir("img_tresh/")
    for ind,file in enumerate(files):
        im = Image.open(file)
        st.image(im)
        opt = SimpleNamespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.5, device='CPU', dnn=False, exist_ok=False,\
                                   half=False, hide_conf=False, hide_labels=False, imgsz=[640, 640], iou_thres=0.45, line_thickness=3,\
                                   max_det=1000, name='exp', nosave=False, project=f'runs/detect{ind}', save_conf=False, save_crop=False,\
                                   save_txt=False,source=file, update=False, view_img=False, visualize=False, weights=['best.pt'])


        det.main(opt)
        imgg = Image.open(f'runs/detect{ind}/exp/{file}')
        st.image(imgg)



