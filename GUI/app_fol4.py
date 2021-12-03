import streamlit as st
from streamlit_folium import folium_static
import folium
from folium import  FeatureGroup
import pandas as pd
from branca.element import Figure
import webbrowser

import base64
import os

from folium import IFrame
from folium.plugins import FloatImage

from PIL import Image
import numpy as np 
import streamlit as st 
import pandas as pd
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf


import json
import requests
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


fig2=Figure(width=300,height=400)
m1 = folium.Map(location=[55.796134, 49.106405], zoom_start=7).add_to(fig2)
#fig2.add_child(m1)
folium.TileLayer('Stamen Terrain').add_to(m1)
folium.TileLayer('Stamen Toner').add_to(m1)
folium.TileLayer('Stamen Water Color').add_to(m1)
folium.TileLayer('cartodbpositron').add_to(m1)
folium.TileLayer('cartodbdark_matter').add_to(m1)

l = ["red", "red", "green", "red", "green", "green", "green", "green", "green", "green", 
"red", "green", "red", "green", "green", "green", "green" ]

data2 = data[data["Type"] == "ТКО"]
data3 = data[(data["Type"] == "БГ") | (data["Type"] == "БД")].reset_index(drop = True)

ll2 = ["gray"] * len(data3)
feature_group1 = FeatureGroup(name='Камеры без превышений')
feature_group2 = FeatureGroup(name='Камеры с обнаруженными превышениями')
feature_group3 = FeatureGroup(name='Другие камеры')

# для загрузки изображения в popup

#file =  '1.PNG'
#dir_base = os.getcwd()
#Filename = dir_base + "\\" + file

#Filename = "0.jpg"

#encoded = base64.b64encode(open(Filename, 'rb').read())

#svg = """
#    <object data="data:image/png;base64,{}" width="{}" height="{} type="image/svg+xml">
#    </object>""".format
#width, height, fat_wh = 516, 570, 0.8
#iframe = IFrame(svg(encoded.decode('UTF-8'), width, height) , width=width*fat_wh, height=height*fat_wh)
#popup  = folium.Popup(iframe,parse_html = True, max_width=2650)



#iframe = IFrame(html = html, width = 500, height = 300) 
#popup = folium.Popup(iframe, max_width = 2650)




for i, val in enumerate(l):
    if val == "green":
        tooltip = data2["Адрес установки камеры"][i]
        folium.Marker( [data2.iloc[i,-2], data2.iloc[i,-1]], 
        popup=data2["Адрес установки камеры"][i], tooltip=tooltip, 
        icon=folium.Icon(color=l[i],icon='ok-sign')).add_to(feature_group1)
    else:
        tooltip = data["Адрес установки камеры"][i]
        folium.Marker( [data.iloc[i,-2], data2.iloc[i,-1]], 
        popup=data["Адрес установки камеры"][i], tooltip=tooltip, 
        icon=folium.Icon(color=l[i],icon='info-sign')).add_to(feature_group2)
        folium.Circle(radius=200,
        location= [data2.iloc[i,-2], data2.iloc[i,-1]], 
        popup='Выделенная территория',
        color='red', fill=True).add_to(m1)


for i in range(len(ll2)):
    tooltip = data3["Адрес установки камеры"][i]
    folium.Marker( [data3.iloc[i,-2], data3.iloc[i,-1]], 
        popup=data3["Адрес установки камеры"][i], tooltip=tooltip, 
        icon=folium.Icon(color=ll2[i],icon='none')).add_to(feature_group3)


if st.button("Кликните на кнопку, чтобы увидеть аналитику по выбранной камере"):
    tooltip = data["Адрес установки камеры"][1]
    folium.Marker( [data.iloc[1,-2], data2.iloc[1,-1]], 
    popup=webbrowser.open('geo_cam_rai1.htm'), tooltip=tooltip, 
    icon=folium.Icon(color=l[1],icon='info-sign')).add_to(feature_group2)
    
m1.add_child(feature_group1)
m1.add_child(feature_group2)
m1.add_child(feature_group3)
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



