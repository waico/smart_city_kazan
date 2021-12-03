# smart_city_kazan
Implementation of functionality to improve the quality of life of the citizens of the great city of Kazan



## Структура

- [OCR + geo_extract](./OCR + geo_extract) folder 
  - Это применение OCR к доступным картинкам для извлечения времени и адреса. 	
  - обработка исходных данных по связке адресов и папок исходных данных. 
  - пайплайн извлечение ширины и долготы по адресу
- [parser](./parser )  folder - автоматические сохранение картинок по яндексовому запросу, для обогащения датасета.  
- [stream_video](stream_video) folder - интеграция стриминг видео 
- [YOLO5_Train_Inference](yolo5x_train) folder - ноутбуки для обучения и инференса моделей, также содержит директорию weights с весами модели
