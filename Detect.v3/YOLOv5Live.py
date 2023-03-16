import time
import torch
import numpy as np
import cv2
import pafy

#pandas,IPython,yaml(pyyaml),tqdm,matplotlib,seaborn добавлены в файлах вышепредставленных библиотек
'''
Выше находятся библиотеки дополнительного импорта (Важно при компиляции)
'''

class ObjectDetection:
    """
    Класс реализует модель Yolo 5 для создания выводов на видео YouTube с использованием OpenCV.
    """
    
    def __init__(self):
        """
        Инициализирует класс с помощью URL-адреса youtube и выходного файла.
        :param url: Должен быть как URL-адрес youtube, по которому делается прогноз.
        :param out_file: Допустимое имя выходного файла.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)



    def load_model(self):
        """
        Загружает модель Yolo5 из концентратора pytorch.
        :return: Обученная модель Pytorch.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True) #s
        return model


    def score_frame(self, frame):
        """
        Принимает один кадр в качестве входных данных и оценивает кадр с использованием модели yolo 5.
        :param frame: входной фрейм в формате numpy/list/tuple.
        :return Метки и координаты объектов, обнаруженных моделью в кадре.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        """
        Для заданного значения метки возвращает соответствующую строку label.
        :param x: числовая метка
        :return: соответствующая метка строки
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        Принимает фрейм и его результаты в качестве входных данных и наносит на фрейм ограничивающие рамки и метку.
        :param results: содержит метки и координаты, предсказанные моделью на данном кадре.
        :param frame: Кадр, который был засчитан.
        :return: Рамка с нанесенными на нее ограничивающими рамками и надписями.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame


    def __call__(self):
        """
        Эта функция вызывается при выполнении класса, она запускает цикл для чтения видео кадр за кадром,
        и записывает выходные данные в новый файл.
        :return: ничего
        """
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 1)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("detect by vir$i", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Создайте новый объект и выполните.
detection = ObjectDetection()
detection()
