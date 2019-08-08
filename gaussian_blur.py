import cv2
import numpy as np
import os
import math


# Расширим фото на размер половины окна. заполним новые ячейки значением крайних пикселей

def photo_extension(img: np.ndarray, ksize: list) -> np.ndarray:
    height, width = img.shape[0], img.shape[1]

    if len(img.shape) ==2:
        new_img = np.zeros((height+ksize[0]-1, width+ksize[1]-1), dtype = np.int64)
        new_height, new_width = new_img.shape[0], new_img.shape[1]
    else:
        new_img = np.zeros((height+ksize[0]-1, width+ksize[1]-1, img.shape[2]), dtype = np.int64)
        new_height, new_width = new_img.shape[0], new_img.shape[1]



    new_img[ksize[0]//2:new_height-ksize[0]//2, ksize[1]//2:new_width-ksize[1]//2] = img



    new_img[0:ksize[0]//2, ksize[1]//2:new_width-ksize[1]//2] = img[0]

    new_img[ksize[0]//2: new_height - ksize[0]//2, 0:ksize[1]//2] = img[0:height, 0:1]


    new_img[new_height- ksize[0]//2:new_height, ksize[1]//2:new_width-ksize[1]//2] = img[height-1]

    new_img[ksize[0]//2: new_height - ksize[0]//2, new_width-ksize[1]//2:new_width] = img[0:height, width-1:width]


    new_img[0:ksize[0]//2, 0:ksize[1]//2] = img[0,0]
    new_img[0:ksize[0]//2,new_width-ksize[1]//2:new_width] = img[0,width-1]
    new_img[new_height-ksize[0]//2:new_height, 0:ksize[1]//2]= img[height-1,0]
    new_img[new_height-ksize[0]//2:new_height, new_width-ksize[1]//2:new_width]= img[height-1,width-1]

    return new_img


def okruglenie (i: float):
    '''if i% 1 < 0.5:
        return math.floor(i)
    else: return math.ceil(i) '''
    return round(i)

def create_mask (window : np.ndarray, sigma: float):
    mask = np.zeros((window.shape[0], window.shape[1]))
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            mask[i][j] = math.exp(-((i-window.shape[0]//2)**2 + (j-window.shape[1]//2)**2)/ (2*sigma**2)) / (2*sigma**2 * math.pi)
    mask = mask/ np.sum(mask)
    return mask


# поиск медианы на полученном окне со стороной k
def new_pixel_intensity (window : np.ndarray, sigma: float, mask):
    if len(window.shape) ==2:
        new_pixel = 0
        #mask = np.zeros((window.shape[0], window.shape[1]))
        for i in range(window.shape[0]):
            for j in range(window.shape[1]):
                #mask[i][j] = math.exp(-((i-window.shape[0]//2)**2 + (j-window.shape[1]//2)**2)/ (2*sigma**2)) / (2*sigma**2 * math.pi)
                new_pixel += mask[i][j]*window[i][j]
                
        #print(new_pixel)
        return okruglenie(new_pixel)
    else:
        res = []
        #print(window)
        for i in range(window.shape[2]):
            res.append(int(np.median(window[::,::,i])))
        return res


def gaussian_blur(img: np.ndarray, ksize: list, sigma: float = 0)->  np.ndarray:

    if sigma == 0:
        sigma = ksize[0]
    new_img = photo_extension(img,ksize)
    
    mask = create_mask(new_img[0:ksize[0],0:ksize[1]], sigma)

    print(mask)

    res = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        
            n_i = i+ ksize[0]//2
            n_j = j+ksize[1]//2
            res[i][j] = new_pixel_intensity(new_img[n_i-ksize[0]//2:n_i+ksize[0]//2+1, n_j-ksize[1]//2:n_j+ksize[1]//2+1], sigma, mask)

            
        print(i)
    
    return res

if __name__ == '__main__':
    img = cv2.imread('1_1.bmp',0)
    
    res = gaussian_blur(img, (3,3), 1)
    #res = photo_extension(img, (11,11))
    cv2.imwrite('results//res1.jpg', res)

    new_res = cv2.GaussianBlur(img, (3,3), 1)
    cv2.imwrite('results//res_opencv.jpg', new_res)

    test1 =( res - new_res)
    cv2.imwrite('results//test.jpg', test1)
    test2 = new_res - res
    cv2.imwrite('results//test2.jpg', test2)
    print(img[0:4,0:4])
    print(res[0:5,0:5])
    print(new_res[0:5,0:5])
    print(test1[0:7,0:7])

    