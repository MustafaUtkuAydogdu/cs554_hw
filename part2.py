import numpy as np
import cv2
import matplotlib.pyplot as plt


#read images
img1 = cv2.imread("homework_dataset\data_disparity_estimation\plastic\left.png")
img2 = cv2.imread("homework_dataset\data_disparity_estimation\plastic\\right.png")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


#PRİNTİNG THE SHAPES OF THE IMAGES
#print(img1.shape)
#print(img2.shape)



def normalized_cross_correlation(patch1, patch2):

    # Normalize patches with mean of every channel and norm of every channel
    mean1 = (np.mean(patch1, axis=(0,1)))
    mean2 = (np.mean(patch2, axis=(0,1)))

    std1 = np.sqrt(np.sum(np.square(patch1 - mean1), axis=(0,1)))

    # if any of the stds are 0, set them to 1e-7
    if std1[0] == 0.0:
        std1[0] = 1e-10
    
    if std1[1] == 0.0:
        std1[1] = 1e-10
    
    if std1[2] == 0.0:
        std1[2] = 1e-10

    std2 = np.sqrt(np.sum(np.square(patch2 - mean2), axis=(0,1)))


    # if any of the stds are 0, set them to 1e-7
    if std2[0] == 0.0:
        std2[0] = 1e-10
    
    if std2[1] == 0.0:
        std2[1] = 1e-10
    
    if std2[2] == 0.0:
        std2[2] = 1e-10

    patch1_normalized = (patch1 - mean1) / std1 
    patch2_normalized = (patch2 - mean2) / std2

    #flatten the patches
    patch1_flatten = patch1_normalized.flatten()
    patch2_flatten = patch2_normalized.flatten()

    # Compute normalized cross-correlation
    ncc = np.sum(patch1_flatten * patch2_flatten) # the higher the better

    return ncc

def disparity_map_creater(img1, img2, patch_size): # img1 is the left image, img2 is the right image

    #get image size
    height, width = img1.shape[:2]
    #pad size for the image
    pad_size = patch_size // 2
    #create disparity map, which is the same size as the left image
    disparity_map = np.zeros(img1.shape)
    #print(disparity_map.shape)


    # pad the images on all sides
    img1 = np.pad(img1, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')
    img2 = np.pad(img2, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')

    #print new shapes to check ( looks good)
    print(img1.shape)
    print(img2.shape)

    # i is the row number, j is the column number
    for i in range(pad_size, height - 2*pad_size ):

        for j in range(pad_size, width - 2*pad_size):
        
            current_patch1 = img1[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, :]
            lookup_range = 65
            #PRİNTİNG THE SHAPES OF THE PATCHES
            #print("patch1: ",current_patch1.shape)
            #initialize max score
            max_score = -99999999
            lookup_range = 65
            lookup_range_limit = j - pad_size
            cur_disparity = 0 
            #search in right image 
            #only within distance of 65 pixels
            for k in range(0, lookup_range+1):
                #if we are out of bounds, break
                if k > lookup_range_limit:
                    break

                current_patch_2 = img2[i - pad_size:i + pad_size + 1, (j - k - pad_size) :j - k + pad_size + 1, :]
                # compute normalized cross-correlation
                current_score = normalized_cross_correlation(current_patch1, current_patch_2)
                # update max_score and min_disparity
                if current_score > max_score:
                    max_score = current_score
                    cur_disparity = k
            disparity_map[ i - pad_size , j - pad_size  ] = cur_disparity



        print("i:", i , "out of", height - pad_size - 1)



    return disparity_map


# test the function


# disparity_map = disparity_map_creater(img1, img2, 5)

# to_save = disparity_map / np.max(disparity_map) * 255
# to_save = to_save.astype(np.uint8)
# cv2.imwrite("disparity_map.png", to_save)


# plt.imshow((disparity_map) / np.max(disparity_map), cmap="gray")
# plt.show()


img1 = cv2.imread("homework_dataset\data_disparity_estimation\cloth\left.png")
img2 = cv2.imread("homework_dataset\data_disparity_estimation\cloth\\right.png")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


disparity_map_2 = disparity_map_creater(img1, img2, 5)
to_save = disparity_map_2 / np.max(disparity_map_2) * 255
to_save = to_save.astype(np.uint8)
cv2.imwrite("disparity_map_2.png", to_save)

plt.imshow((disparity_map_2) / np.max(disparity_map_2), cmap="gray")

plt.show()