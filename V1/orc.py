from PIL import Image
import pytesseract
import cv2
import os
import csv

def search_letter(text1):
    for company in companes:
        for j in range(1,len(text1)//3):
            if text1[3*(j-1):3*j] in company:
                return company, True
    return None, False


file_table = open("/media/sf_shared/MAILHACK/train.csv","w")
train_table_w = csv.writer(file_table)
main_directory = "/media/sf_shared/MAILHACK/train"
not_main_directory = os.listdir(main_directory)
count = 0
count_true = 0
total_count = 0
companes = []
for company in not_main_directory:
    companes += [company[company.find(" ")+1:]]


    files = os.listdir("/media/sf_shared/MAILHACK/test_new/" )

    for j in range(len(files)):

        flag = True
        image = files[j]
        if image[image.find(".")+1:] == "jpg" or image[image.find(".")+1:] == "png":
            image_path = "/media/sf_shared/MAILHACK/test_new/" +image
            image_path1 = "/media/sf_shared/MAILHACK/test_new/"  + image

            try:
                image_path1 = cv2.imread(image_path1)
                gray = cv2.cvtColor(image_path1, cv2.COLOR_BGR2GRAY)
            except:
                flag = False
            preprocess = "thresh"
            total_count += 1

            if flag :
                # загрузить образ и преобразовать его в оттенки серого
                image_path = cv2.imread(image_path)

                # проверьте, следует ли применять пороговое значение для предварительной обработки изображения

                if preprocess == "thresh":
                    gray = cv2.threshold(gray, 0, 255,
                                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                # если нужно медианное размытие, чтобы удалить шум
                elif preprocess == "blur":
                    gray = cv2.medianBlur(gray, 3)

                # сохраним временную картинку в оттенках серого, чтобы можно было применить к ней OCR

                filename = "{}.png".format(os.getpid())
                cv2.imwrite(filename, gray)

                # загрузка изображения в виде объекта image Pillow, применение OCR, а затем удаление временного файла
                text = pytesseract.image_to_string(Image.open(filename))
                answer = search_letter(text)
                if answer[1]:
                    train_table_w.writerow([answer[0], image])
                else:
                    train_table_w.writerow(["other",image])
