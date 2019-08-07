import cv2
import numpy as np
import os


# Расширим фото на размер половины окна. заполним новые ячейки значением крайних пикселей

def photo_extension(img: np.ndarray, k: int) -> np.ndarray:
    height, width = img.shape[0], img.shape[1]

    if len(img.shape) ==2:
        new_img = np.zeros((height+k-1, width+k-1), dtype = np.int64)
        new_height, new_width = new_img.shape[0], new_img.shape[1]
    else:
        new_img = np.zeros((height+k-1, width+k-1, img.shape[2]), dtype = np.int64)
        new_height, new_width = new_img.shape[0], new_img.shape[1]



    new_img[k//2:new_height-k//2, k//2:new_width-k//2] = img



    new_img[0:k//2, k//2:new_width-k//2] = img[0]

    new_img[k//2: new_height - k//2, 0:k//2] = img[0:height, 0:1]


    new_img[new_height- k//2:new_height, k//2:new_width-k//2] = img[height-1]

    new_img[k//2: new_height - k//2, new_width-k//2:new_width] = img[0:height, width-1:width]


    new_img[0:k//2, 0:k//2] = img[0,0]
    new_img[0:k//2,new_width-k//2:new_width] = img[0,width-1]
    new_img[new_height-k//2:new_height, 0:k//2]= img[height-1,0]
    new_img[new_height-k//2:new_height, new_width-k//2:new_width]= img[height-1,width-1]

    return new_img


'''
def medianblur_gray (img: np.ndarray, k: int) -> np.ndarray:
    

    
    return 

'''

if __name__ == '__main__':
    img = cv2.imread('test_rgb.jpg')
    res = photo_extension(img, 51)

    cv2.imwrite('results//res1.jpg', res)