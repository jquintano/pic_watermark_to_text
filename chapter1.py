import cv2
from func import txtbox, txtget
import os

path = r'C:\\Users\\WallyIM\\Desktop\\PY\\OpenCV\\Resources'
dir = os.listdir(path)
for file in dir:
    print(file)
    if file[-3:] == 'jpg':
        try:
            boxed = txtbox("Resources/"+file)
            boxedHSV = cv2.cvtColor(boxed, cv2.COLOR_BGR2HSV)
            boxedBlur = cv2.GaussianBlur(boxed, (5, 5), 1)
            txt = txtget(boxedHSV)

        # print(f"{txt} == {file}")
            cv2.imshow("boxedHSV", boxedHSV)
            f_name, f_ext = os.path.splitext(file)
            new_name = '{}{}'.format(txt.strip(), f_ext)
            os.rename("Resources/"+file, new_name)
            print(new_name)
            print(txt, len(txt))
            cv2.waitKey(0)
            break
        except:
            continue






