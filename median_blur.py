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


# поиск медианы на полученном окне со стороной k
def find_median (window : np.ndarray):
    if len(window.shape) ==2:
        return int(np.median(window))
    else:
        res = []
        #print(window)
        for i in range(window.shape[2]):
            res.append(int(np.median(window[::,::,i])))
        return res


def median_blur(img: np.ndarray, k: int)->  np.ndarray:
    new_img = photo_extension(img,k)
    res = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        
            n_i = i+ k//2
            n_j = j+k//2
            res[i][j] = find_median(new_img[n_i-k//2:n_i+k//2+1, n_j-k//2:n_j+k//2+1])             
        print(i)
    return res

if __name__ == '__main__':
    img = cv2.imread('test_rgb.jpg')
    res = median_blur(img, 21)
    cv2.imwrite('results//res1.jpg', res)

    new_res = cv2.medianBlur(img, 21)
    cv2.imwrite('results//res_opencv.jpg', new_res)

    test = res - new_res
    cv2.imwrite('results//test.jpg', test)