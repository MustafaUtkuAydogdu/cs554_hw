import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def get_sift_features_and_descriptors(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #initalize the sift feature detector
    sift = cv2.SIFT_create()

    #get the keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return keypoints, descriptors



def display_sift_features(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #initalize the sift feature detector
    sift = cv2.SIFT_create()

    #get the keypoints and descriptors
    keypoints, _ = sift.detectAndCompute(gray_image, None)

    #draw the keypoints on the image
    image = cv2.drawKeypoints(image, keypoints, None)

    #show the image
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# test display_sift_features
#display_sift_features("homework_dataset\data_image_stitching\im1_1.png")

# test get_sift_features_and_descriptors
# keypoints, descriptors = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im1_1.png")
# print("Number of keypoints Detected: ", len(keypoints))
# print("Descriptor Shape: ", descriptors.shape) # (keypoints, 128)


#we'LL use brute force to match the descriptors of the two images
#based on the eucledian distance between the descriptors
def match_features(descriptors1, descriptors2):

    #hold a list of pairs of matching descriptors
    matches = []
    for i in range(len(descriptors1)):
        current_descriptor = descriptors1[i]

        #find the eucledian distance between the descriptor and all descriptors in descriptors2
        eucledian_distances = np.linalg.norm(descriptors2 - current_descriptor, axis=1)

        #sort the eucledian distances
        indices = np.argsort(eucledian_distances)
        closest_descriptor_index = indices[0] 
        second_closest_descriptor_index = indices[1]

        #we found a match, save pair to matches
        match = [i, closest_descriptor_index]
        # matches.append(match)

        #check if the smallest distance is less than 0.75 times the second smallest distance
        if eucledian_distances[closest_descriptor_index] < 0.75 * eucledian_distances[second_closest_descriptor_index]:
            match.append(eucledian_distances[closest_descriptor_index])
            matches.append(match)

    return matches


#test match_features
# keypoints1, descriptors1 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im2_1.jpg")
# keypoints2, descriptors2 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im2_2.jpg")
# matches = match_features(descriptors1, descriptors2)

# print("Number of matches: ", len(matches))

# get the matching keypoints for each of the images
keypoints_matched = []
# keypoints_matched_2 = []

def derive_points(keypoints1,keypoints2, matches):
    img_points_1 = np.float32([keypoints1[m[0]].pt for m in matches])
    img_points_1 = img_points_1.reshape(-1, 1, 2)
    img_points_2 = np.float32([keypoints2[m[1]].pt for m in matches])
    img_points_2 = img_points_2.reshape(-1, 1, 2)

    return img_points_1 , img_points_2


# first_img_points = np.float32([keypoints1[m[0]].pt for m in matches])
# first_img_points = first_img_points.reshape(-1, 1, 2)
# second_img_points = np.float32([keypoints2[m[1]].pt for m in matches])
# second_img_points = second_img_points.reshape(-1, 1, 2)


def keypoints_matched_to_list(first_img_points,second_img_points):
    keypoints_matched_list = []
    for i in range(len(first_img_points)):
        first = first_img_points[i]
        second = second_img_points[i]
        keypoints_matched_list.append([first[0][0], first[0][1], second[0][0], second[0][1]])
    
    return keypoints_matched_list


# first_img_points , second_img_points = derive_points(keypoints1,keypoints2, matches)
# keypoints_matched = keypoints_matched_to_list(first_img_points,second_img_points)


#print("Number of keypoints_matched: ", len(keypoints_matched))


#define a function that displays the matches
def display_matches(img1_path, img2_path, keypoints_matched):
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)

    #draw the matches
    for match in keypoints_matched:
        x1, y1, x2, y2 = match[0], match[1], match[2], match[3]
        cv2.circle(image1, (int(x1), int(y1)), 3, (0, 0, 255), -1)
        cv2.circle(image2, (int(x2), int(y2)), 3, (0, 0, 255), -1)
        #cv2.line(image1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    #display the images with ratio 0.8
    scale_percent = 80
    width1 = int(image1.shape[1] * scale_percent / 100)
    height1 = int(image1.shape[0] * scale_percent / 100)
    width2 = int(image2.shape[1] * scale_percent / 100)
    height2 = int(image2.shape[0] * scale_percent / 100)
    dim1 = (width1, height1)
    dim2 = (width2, height2)
    resized_image1 = cv2.resize(image1, dim1, interpolation = cv2.INTER_AREA)
    resized_image2 = cv2.resize(image2, dim2, interpolation = cv2.INTER_AREA)

    #show the images
    cv2.imshow("image1", resized_image1)
    cv2.imshow("image2", resized_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#display_matches("homework_dataset\data_image_stitching\im2_1.jpg", "homework_dataset\data_image_stitching\im2_2.jpg", keypoints_matched[:5])



def prepare_matrix_for_homography(keypoints_matched):
    #concate the keypoints into a matrix
    X = []
    for x1, y1 , x2, y2 in keypoints_matched:
        X.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        X.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

    return X

# write a function to compute the homography between two images
def compute_homography(X):

    # do the eigenvector analysis
    _, _, eigenvectors = np.linalg.svd(X)

    #reshape the eigenvector corresponding to the smallest eigenvalue into the desired homography matrix, and normalize it
    smallest_eigenvector = eigenvectors[-1, :]
    homography = smallest_eigenvector.reshape(3, 3) / eigenvectors[-1, -1] # normalize by the last element of the smallest eigenvector

    #print("norm_factor : ",eigenvectors[-1,-1])
    
    #print(eigenvectors)

    
    return homography



# # test compute_homography
# X = prepare_matrix_for_homography(keypoints_matched)
# homography = compute_homography(X)
# print("Homography Matrix: ", homography)


def ransac(keypoints_matched, threshold=1000):

    best_num_inliers = 0
    best_homography = []
    best_inliers = []
    num_iters = 1000
    min_error = 1000000

    #iterate through all the matches
    for i in range(num_iters):

        # chose random half of the matches
        num_matches = len(keypoints_matched)
        random_indices = np.random.choice(num_matches, 4, replace=False)
        # get the random matches
        hypothetical_inliers = []
        for index in random_indices:
            hypothetical_inliers.append(keypoints_matched[index])

        # get the homography matrix
        X = prepare_matrix_for_homography(hypothetical_inliers)
        homography = compute_homography(X)

        # reset the current_inliers for the next iteration
        cur_num_inliers = 0
        current_inliers = []
        cur_error = 0
        #iterate through all the matches
        for x1,y1,x2,y2 in (keypoints_matched):

            #compute the current error , using homogeneous coordinates
            homegenous_p1 = np.array([[x1], [y1], [1]])
            homegenous_p2 = np.array([[x2], [y2], [1]])
            #print shapes
            # print("homegenous_p1 : ",homegenous_p1.shape)
            # print("homegenous_p2 : ",homegenous_p2.shape)
            # print("homography : ",homography.shape)
            p1_transformed = np.matmul(homography,homegenous_p1)

            #normalize the transformed point
            p1_transformed = p1_transformed / p1_transformed[2]


            current_difference = (homegenous_p2) - p1_transformed 

            # print("p1_transformed : ",p1_transformed.shape)
            # print("current_difference : ",current_difference.shape)

            #get squared error between the transformed point and the actual point
            res = sum(current_difference**2)
            #check if the error is within the threshold
            if res < threshold:
                cur_num_inliers += 1
                current_inliers.append([x1, y1, x2, y2])
                cur_error += res

        # check if the current iteration is the best iteration
        if cur_num_inliers > best_num_inliers:
            best_num_inliers = cur_num_inliers
            best_homography = homography
            best_inliers = current_inliers
            min_error = cur_error

    return best_num_inliers, best_homography, best_inliers , min_error


## test ransac
# best_num_inliers, best_homography , best_inliers, min_error  = ransac(keypoints_matched,threshold=1000)

# #recalculate the homography using the best inliers
# X = prepare_matrix_for_homography(best_inliers)
# best_homography = compute_homography(X)




# print("Best Homography Matrix: ", best_homography)
# print("Best Number of Inliers: ", best_num_inliers)
# print("Min error : ",min_error)

# # Example usage
# img1 = cv2.imread("homework_dataset\data_image_stitching\im1_1.png")
# img2 = cv2.imread('homework_dataset\data_image_stitching\im1_2.png')


def image_to_matrix(image):

    height, width, num_channels = image.shape
    matrix = np.zeros((width, height, num_channels), dtype='int')

    for i in range(height):
        matrix[:, i] = image[i]

    return matrix

def matrix_to_image(matrix):
    height, width, num_channels = matrix.shape
    image = np.zeros((width, height, num_channels), dtype='int')


    for i in range(height):
        image[:, i] = matrix[i]

    return image


def warpImage(image, homography, dim):
    row_count = dim[0]
    column_count = dim[1]


    #m_image = image_to_matrix(image)

    height, width, num_channels = image.shape
    matrix = np.zeros((width, height, num_channels), dtype='int')

    for i in range(height):
        matrix[:, i] = image[i]
    
    m_image = matrix
    warped_image = np.zeros((row_count, column_count, m_image.shape[2]), dtype='float32')

    print("r : ",row_count," c : ",column_count)

    # #hold the corner points of the image in one
    # left_top = [0, 0]
    # right_top = [0, c - 1]
    # left_bottom = [r - 1, 0]
    # right_bottom = [r - 1, c - 1]
    # corners = np.array([left_top, right_top, left_bottom, right_bottom])
    # #get the transformed corner points
    # transformed_corners = np.dot(homography, corners.T).T
    # transformed_corners = transformed_corners / transformed_corners[:, 2].reshape(-1, 1)


    for i in range(m_image.shape[0]):
        for j in range(m_image.shape[1]):
            homogenous_coordinates = np.dot(homography, [i, j, 1])
            #normalize
            homogenous_coordinates = homogenous_coordinates / homogenous_coordinates[2]
            #print("mapped_point : ",mapped_point[2])
            x, y, _ = (homogenous_coordinates).astype(int)

            limit_check_1 = True if x < row_count and x >= 0 else False
            limit_check_2 = True if y < column_count and y >= 0 else False

            if limit_check_1 and limit_check_2:
                warped_image[x, y] = m_image[i, j]


    # hold tmp to prevent overwriting
    tmp = warped_image.copy()
    #interpolate the missing pixels - kernel 3x3
    for i in range(warped_image.shape[0]):
        for j in range(warped_image.shape[1]):
            if tmp[i, j].all() == 0:
                warped_image[i, j] =  get_average_pixel(tmp, i, j, 2)

    # # #interpolate the missing pixels
    # for i in range(m_dest_image.shape[0]):
    #     for j in range(m_dest_image.shape[1]):
    #         if m_dest_image[i, j].all() == 0:
    #             m_dest_image[i, j] =  m_dest_image[i-1, j-1]  
    

    height, width, num_channels = warped_image.shape
    image = np.zeros((width, height, num_channels), dtype='int')

    for i in range(height):
        image[:, i] = warped_image[i]

    warped_image = image

    return warped_image


#define a function that checks nxn pixels around a pixel and returns the average of the pixels
def get_average_pixel(image, i, j, n):
    #get the dimensions of the image
    rows, cols, depth = image.shape

    #get the start and end points of the nxn window
    start_row = max(0, i - n)
    end_row = min(rows, i + n + 1)
    start_col = max(0, j - n)
    end_col = min(cols, j + n + 1)

    #get the number of non zero pixels in the window
    num_non_zero_pixels = np.count_nonzero(image[start_row:end_row, start_col:end_col], axis=(0, 1))

    #print(num_non_zero_pixels)
    # single_num = sum(num_non_zero_pixels)
    # # if majority of the pixels are zero, return zero
    # if single_num < 3*n*n/ (2) :
    #     return np.array([0,0,0])
    
    #get the average of the pixels
    if  num_non_zero_pixels.all() != 0:
        avg_pixel = np.sum(image[start_row:end_row, start_col:end_col], axis=(0, 1)) / num_non_zero_pixels
    else:
        avg_pixel = np.array([0,0,0])

    return avg_pixel


def alignImages(left_img, warped_image):

    height, width, num_channels = left_img.shape
    matrix = np.zeros((width, height, num_channels), dtype='int')
    for i in range(height):
        matrix[:, i] = left_img[i]
    left_img = matrix
    
    height, width, num_channels = warped_image.shape
    matrix = np.zeros((width, height, num_channels), dtype='int')
    for i in range(height):
        matrix[:, i] = warped_image[i]
    warped_image = matrix

    print("m_img_to_copy : ",left_img.shape[0]," ",left_img.shape[1])
    print("m_dest_img : ",warped_image.shape[0]," ",warped_image.shape[1])

    for i in range(left_img.shape[0]):
        for j in range(left_img.shape[1]):
            if warped_image[i, j].all() == 0:
                warped_image[i, j] = left_img[i, j]
            else:
                alpha = min((j/left_img.shape[1]) + 0.5, 1)
                warped_image[i, j] = (1-alpha)*left_img[i, j] + alpha*warped_image[i, j]


    height, width, num_channels = warped_image.shape
    image = np.zeros((width, height, num_channels), dtype='int')


    for i in range(height):
        image[:, i] = warped_image[i]

    warped_image = image
    return (warped_image)

def alpha_blending(alpha, pix1, pix2):

    return (1-alpha)*pix1 + alpha*pix2



#get the dimensions of the images

#--------For pair 1
# h1, w1, _ = img1.shape
# h2, w2, _ = img2.shape




# result_my = warpImage(img2, np.linalg.inv(best_homography), (w1 + w2, max(h1, h2)))
# warped = result_my.astype(np.float32)/255
# import matplotlib.pyplot as plt
# plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.show()



# result_my = alignImages(img1, result_my)
# float_img = result_my.astype(np.float32)/255
# import matplotlib.pyplot as plt
# plt.imshow(cv2.cvtColor(float_img, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.show()

#--------For pair 2

# compare our ransac with opencv's findHomography
# opencv_best_homography, _ = cv2.findHomography(first_img_points, second_img_points, cv2.RANSAC, 5.0)
# print("OpenCV's Best Homography Matrix: ", opencv_best_homography)



# iterate through points to see whether transform is correct : YES!Correct
# for i in range(len(first_img_points)):
#     src_pt = first_img_points[i]
#     dst_pt = second_img_points[i]
#     src_pt = np.array([src_pt[0][0], src_pt[0][1], 1])
#     dst_pt = np.array([dst_pt[0][0], dst_pt[0][1], 1])
#     print(np.matmul(best_homography, src_pt) / np.matmul(best_homography, src_pt)[2])
#     print(dst_pt)
#     print("")
#     print("")


# 4 images stitching

# img1 = cv2.imread("homework_dataset\data_image_stitching\im2_1.jpg")
# img2 = cv2.imread('homework_dataset\data_image_stitching\im2_2.jpg')
# img3 = cv2.imread('homework_dataset\data_image_stitching\im2_3.jpg')
# img4 = cv2.imread('homework_dataset\data_image_stitching\im2_4.jpg')




# keypoints1, descriptors1 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im2_1.jpg")
# keypoints2, descriptors2 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im2_2.jpg")

# matches1 = match_features(descriptors1, descriptors2)


# first_img_points1 , second_img_points1 = derive_points(keypoints1,keypoints2, matches1)
# keypoints_matched1 = keypoints_matched_to_list(first_img_points1,second_img_points1)

# best_num_inliers1, best_homography1 , best_inliers1, min_error1 = ransac(keypoints_matched1,threshold=1000)
# X1 = prepare_matrix_for_homography(best_inliers1)
# best_homography1 = compute_homography(X1)
# print("min_error1 : ",min_error1)


# # warp image 2
# h1, w1, _ = img1.shape
# h2, w2, _ = img2.shape

# warped_image1 = warpImage(img2, np.linalg.inv(best_homography1), (w1 + w2, max(h1, h2)))
# warped1 = warped_image1.astype(np.float32)/255


# import matplotlib.pyplot as plt
# plt.imshow(cv2.cvtColor(warped1, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.show()


# #align image 2
# aligned_image_1 = alignImages(img1, warped_image1)
# aligned_image1 = aligned_image_1.astype(np.float32)/255

# plt.imshow(cv2.cvtColor(aligned_image1, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.show()



# # save aligned image 1
# cv2.imwrite("aligned_image1.jpg",aligned_image1*255)



















# keypoints3, descriptors3 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im2_3.jpg")
# keypoints4, descriptors4 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im2_4.jpg")

# matches2 = match_features(descriptors3, descriptors4)


# first_img_points2 , second_img_points2 = derive_points(keypoints3,keypoints4, matches2)
# keypoints_matched2 = keypoints_matched_to_list(first_img_points2,second_img_points2)

# best_num_inliers2, best_homography2 , best_inliers2, min_error2  = ransac(keypoints_matched2,threshold=1000)
# X2 = prepare_matrix_for_homography(best_inliers2)
# best_homography1 = compute_homography(X2)
# print("min_error2 : ",min_error2)

# # warp image 2
# h3, w3, _ = img3.shape
# h4, w4, _ = img4.shape

# warped_image2 = warpImage(img4, np.linalg.inv(best_homography1), (w3 + w4, max(h3, h4)))
# warped2 = warped_image2.astype(np.float32)/255


# import matplotlib.pyplot as plt
# plt.imshow(cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.show()


# #align image 2
# aligned_image_2 = alignImages(img3, warped_image2)
# aligned_image2 = aligned_image_2.astype(np.float32)/255

# plt.imshow(cv2.cvtColor(aligned_image2, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.show()


# #save aligned image 2
# cv2.imwrite("aligned_image2.jpg",aligned_image2*255)


#------------------------------------------
#4 images working -start

# print("heeeeeeey")
# #repeat the process for the last two images
# img5 = cv2.imread("aligned_image1.jpg")
# img6 = cv2.imread('aligned_image2.jpg')



# keypoints3, descriptors3 = get_sift_features_and_descriptors("aligned_image1.jpg")
# keypoints4, descriptors4 = get_sift_features_and_descriptors("aligned_image2.jpg")


# matches3 = match_features(descriptors3, descriptors4)


# first_img_points3 , second_img_points3 = derive_points(keypoints3,keypoints4, matches3)
# keypoints_matched3 = keypoints_matched_to_list(first_img_points3,second_img_points3)

# best_num_inliers3, best_homography3 , best_inliers3, min_error3  = ransac(keypoints_matched3,threshold=1000)
# X3 = prepare_matrix_for_homography(best_inliers3)
# best_homography3 = compute_homography(X3)
# print("min_error3 : ",min_error3)


# # warp image 2
# h1, w1, _ = img5.shape
# h2, w2, _ = img6.shape

# warped_image3 = warpImage(img6, np.linalg.inv(best_homography3), (w1 + w2, max(h1, h2)))
# warped3 = warped_image3.astype(np.float32)/255

# import matplotlib.pyplot as plt

# plt.imshow(cv2.cvtColor(warped3, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.show()

# #align image 2
# aligned_image_3 = alignImages(img5, warped_image3)
# aligned_image3 = aligned_image_3.astype(np.float32)/255

# plt.imshow(cv2.cvtColor(aligned_image3, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.show()


#------------------------------------------
#4 images working -end

#------------------------------------------
#First two images working -start
# test on 2 images
img1 = cv2.imread("homework_dataset\data_image_stitching\im1_1.png")
img2 = cv2.imread('homework_dataset\data_image_stitching\im1_2.png')

keypoints1, descriptors1 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im1_1.png")
keypoints2, descriptors2 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im1_2.png")

matches1 = match_features(descriptors1, descriptors2)

first_img_points1 , second_img_points1 = derive_points(keypoints1,keypoints2, matches1)

keypoints_matched1 = keypoints_matched_to_list(first_img_points1,second_img_points1)

best_num_inliers1, best_homography1 , best_inliers1, min_error1  = ransac(keypoints_matched1,threshold=1000)


X1 = prepare_matrix_for_homography(best_inliers1)

best_homography1 = compute_homography(X1)

print("min_error1 : ",min_error1)


# warp image 2
h1, w1, _ = img1.shape
h2, w2, _ = img2.shape

warped_image1 = warpImage(img2, np.linalg.inv(best_homography1), (w1 + w2, max(h1, h2)))
warped1 = warped_image1.astype(np.float32)/255


plt.imshow(cv2.cvtColor(warped1, cv2.COLOR_BGR2RGB), aspect='auto')
plt.show()

#align image 2

aligned_image_1 = alignImages(img1, warped_image1)
aligned_image1 = aligned_image_1.astype(np.float32)/255
plt.imshow(cv2.cvtColor(aligned_image1, cv2.COLOR_BGR2RGB), aspect='auto')
plt.show()

#------------------------------------------
#First two images working -end


#------------------------------------------
#Last two images working -start

img1 = cv2.imread("homework_dataset\data_image_stitching\im3_0.jpg")
img2 = cv2.imread('homework_dataset\data_image_stitching\im3_1.jpg')

keypoints1, descriptors1 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im3_0.jpg")

keypoints2, descriptors2 = get_sift_features_and_descriptors("homework_dataset\data_image_stitching\im3_1.jpg")

matches1 = match_features(descriptors1, descriptors2)

first_img_points1 , second_img_points1 = derive_points(keypoints1,keypoints2, matches1)

keypoints_matched1 = keypoints_matched_to_list(first_img_points1,second_img_points1)

best_num_inliers1, best_homography1 , best_inliers1, min_error1  = ransac(keypoints_matched1,threshold=1000)


X1 = prepare_matrix_for_homography(best_inliers1)

best_homography1 = compute_homography(X1)

print("min_error1 : ",min_error1)


# warp image 2
h1, w1, _ = img1.shape
h2, w2, _ = img2.shape

warped_image1 = warpImage(img2, np.linalg.inv(best_homography1), (w1 + w2, max(h1, h2)))
warped1 = warped_image1.astype(np.float32)/255


plt.imshow(cv2.cvtColor(warped1, cv2.COLOR_BGR2RGB), aspect='auto')
plt.show()

#align image 2
aligned_image_1 = alignImages(img1, warped_image1)
aligned_image1 = aligned_image_1.astype(np.float32)/255
plt.imshow(cv2.cvtColor(aligned_image1, cv2.COLOR_BGR2RGB), aspect='auto')
plt.show()

