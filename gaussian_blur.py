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
    #  if i% 1 < 0.5:
    #      return math.floor(i)
    #  else: return math.ceil(i)
    return round(i)

def create_mask (window : np.ndarray, sigma: float):
    mask = np.zeros((window.shape[0], window.shape[1]))
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            mask[i][j] = math.exp(-((i-window.shape[0]//2)**2 + (j-window.shape[1]//2)**2)/ (2*sigma**2)) / (2*sigma**2 * math.pi)
    mask = mask/ np.sum(mask)
    return mask


# поиск медианы на полученном окне со стороной k
def pixel_intensity (window : np.ndarray, sigma: float, mask):
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
    
    mask = new_create_mask(ksize[0], ksize[1], sigma,sigma)

    print(np.round(mask/mask[0][0]))   

    flag = False
    if len(img.shape) == 3:
        mask = color_mask(mask)
        flag = True

    res = np.copy(img)
    print(res.shape, '    ', mask.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        
            n_i = i+ ksize[0]//2
            n_j = j+ksize[1]//2
            res[i][j] = new_pixel_intensity(new_img[n_i-ksize[0]//2:n_i+ksize[0]//2+1, n_j-ksize[1]//2:n_j+ksize[1]//2+1], mask, flag)
           
        
        #print(res[i][j], '  ', img[i][j])
    
    return res

################

def new_pixel_intensity (window : np.ndarray, mask: np.ndarray, flag: bool):
    if flag:
        if len(window.shape) ==3:
            new_pixel = np.sum(np.sum(window*mask, axis = 0), axis= 0)
        else:
            new_pixel = np.sum(window*mask,axis= 0)
    else:
        new_pixel = np.sum(window*mask)

    return np.round(new_pixel)




def new_gaussian_blur(img: np.ndarray, ksize: list, sigma_x: float = 1,sigma_y: float = -1)->  np.ndarray:
    if sigma_y == -1:
        sigma_y = sigma_x
    new_img = photo_extension(img,ksize)
    
    mask_x = new_create_mask_1d(ksize[0], sigma_x)
    mask_y = new_create_mask_1d(ksize[1], sigma_y)



    print('mask_x',mask_x)
    print('mask_y',mask_y)

    flag = False
    if len(img.shape) == 3:
        mask_x = color_mask(mask_x)
        mask_y = color_mask(mask_y)
        flag = True

    res = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        
            n_i = i+ ksize[0]//2
            n_j = j+ksize[1]//2
            res[i][j] = new_pixel_intensity(new_img[n_i-ksize[0]//2:n_i+ksize[0]//2+1, n_j ], mask_x, flag)
            
    
    new_img = photo_extension(res,ksize)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        
            n_i = i+ ksize[0]//2
            n_j = j+ksize[1]//2
            res[i][j] = new_pixel_intensity(new_img[n_i, n_j-ksize[1]//2:n_j+ksize[1]//2+1], mask_y, flag)
            
    
    
    return res



#################################

def color_mask(mask: np.ndarray):
    if len(mask.shape) ==2:
        new_mask = np.zeros((mask.shape[0],mask.shape[1],3))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                a = mask[i,j]
                new_mask[i,j] = [a,a,a]
    else:
        new_mask = np.zeros((mask.shape[0],3))
        for i in range(mask.shape[0]):
                a = mask[i]
                new_mask[i] = [a,a,a]
    return new_mask

def new_create_mask(X: int, Y: int, sigma_x: float, sigma_y: float):
    mask = np.zeros((X, Y))
    for i in range(X):
        for j in range(Y):
            mask[i][j] = mask_pixel(i-X//2,j-Y//2,sigma_x, sigma_y)
    mask = mask/ np.sum(mask)
    return mask


def mask_pixel (x:int, y:int, sigma_x: float, sigma_y: float, step:int = 1000):
    x_l = np.linspace(x-0.5,x+0.5, step)
    y_l = np.linspace(y-0.5, y+0.5, step)
    res = 0
    for i in range(step):
        for j in range(step):
            res += (math.exp(-1*(x_l[i]**2)/(2*sigma_x**2) - (y_l[j]**2)/ (2*sigma_y**2)) / (2*sigma_x*sigma_y * math.pi))
    return res

def new_create_mask_1d(X: int, sigma_x: float):
    mask = np.zeros(X)
    for i in range(X):
        mask[i] = mask_pixel_1d(i-X//2,sigma_x)
    mask = mask/ np.sum(mask)
    return mask



def mask_pixel_1d (x:int, sigma_x: float, step:int = 1000):
    
    x_l = np.linspace(x-0.5,x+0.5, step)
    res = 0
    for i in range(step):
        res += (math.exp(-1*(x_l[i]**2)/(2*sigma_x**2)) / (2*sigma_x**2 * math.pi))
    return res



if __name__ == '__main__':
    img = cv2.imread('1_1.bmp',0)
    
    res = new_gaussian_blur(img, (5,5), 1)
    #res = photo_extension(img, (11,11))
    cv2.imwrite('results//res1.jpg', res)

    new_res = cv2.GaussianBlur(img, (5,5), 1)
    cv2.imwrite('results//res_opencv.jpg', new_res)
    new_img = np.zeros((img.shape[0],img.shape[1], 3), dtype = np.int64)

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if abs(res[i][j] - new_res[i][j]) < 5:
    #             new_img[i][j] = [0,255,0]
    #         if abs(res[i][j] - new_res[i][j]) > 10:
    #             new_img[i][j] = [255,255,255]
    #         if abs(res[i][j] - new_res[i][j]) == 0:
    #             new_img[i][j] = [0,0,0]
    #         if abs(res[i][j] - new_res[i][j]) == 1:
    #             new_img[i][j] = [0,0,255]
    #         if abs(res[i][j] - new_res[i][j]) == 6:
    #             new_img[i][j] = [0,0,255]
            
    test1 =( res - new_res)
    cv2.imwrite('results//test.jpg', new_img)
    test2 = new_res - res
    cv2.imwrite('results//test2.jpg', test2)
    test3 = res - new_res
    cv2.imwrite('results//test3.jpg', test3)
    print(img[0:4,0:4])
    print(res[0:5,0:5])
    print(new_res[0:5,0:5])
    print(test1[0:7,0:7])
    print(new_create_mask_1d(7,1))