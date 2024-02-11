import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def k_means(data,k, max_iter=30, threshold=1e-3):

    L = data.shape[0]
    means = np.sort(rng.choice(L, size=k, replace=False))
    means = np.array([ data[idx] for idx in means])
    means = np.float32(means)
    variance = np.zeros(k)
    data2 = np.reshape(data, (L,1))
    classified = np.argmin( abs(data2 - means), axis=1)
    count = 0
    flag = 0
    means_old = 0 + means
    while ( count<max_iter and flag==0) :
        for idx in range(k):
            means[idx] = np.mean(data[classified==idx])
            variance[idx] = np.std(data[classified==idx])
        classified = np.argmin(abs(data2 - means), axis=1)
        err = np.sum(abs(means_old - means))


        if err < threshold:
            flag =1
            # print('flag changed to 1 after ' + str(count) + ' iterations, with error = ' + str(err))
        count += 1
        means_old = 0 + means

    print(data)
    return classified, variance

# k =3
# NumOfPoints = 12
rng = np.random.default_rng()
#
# data = np.sort(rng.choice(100, size=NumOfPoints, replace=False))
# data = np.array([1,3,6,10,40,45,66,70,102,110,130,200])
# for tries in range(5):
#     classified, variance = k_means(data,3)
#     print(np.sum(variance))
#     print(str(classified) +'\n')


def k_means_3d(data,k, max_iter=30, threshold=1e-3):

    # reshaping data from (x,y,z) to 2D (x*y,z)
    # RGB pixel groups example from (4,4,3) to (16,3)
    data_org = data
    dims = data.shape
    depth = dims[-1]
    L = round(np.linalg.det(np.identity(len(dims[:-1])) * dims[:-1]))
    data = np.reshape(data, (L,dims[-1]))
    Segemented_image = np.zeros(data.shape)

    # randomizes 1st means guess  k-depth size (k,3)
    means = np.sort(rng.choice(L, size=k, replace=False))
    means = np.array([ data[idx,:] for idx in means])
    means = np.float32(means)
    variance = np.zeros(k)

    # data reshape for broadcasting sucusses before diff substraction.
    # norma2 for diff to get scalar distance instead of (3,1) vector
    data2 = np.reshape(data, (L,depth,1))
    diff_norm = np.linalg.norm(data2 - means.T, axis=1)
    classified = np.argmin( diff_norm, axis=1)
    count = 0
    flag = 0
    means_old = 0 + means
    while ( count<max_iter and flag==0) :
        #  means re-evaluation loop
        for idx in range(k):
            means[idx,:] = np.mean(data[classified==idx], axis=0)
            # variance[idx] = np.std(data[classified==idx], axis=0)
        diff_norm = np.linalg.norm(data2 - means.T, axis=1)
        classified = np.argmin(diff_norm, axis=1)
        err = np.sum(abs(means_old - means))
        print(classified)
        print( 'the error is: ' + str(err) + ' \n')


        if err < threshold:
            flag =1
            print('flag changed to 1 after ' + str(count) + ' iterations, with error = ' + str(err))
        count += 1
        means_old = 0 + means
    for idx in range(k):
        Segemented_image[classified==idx] = means[idx]
    Segemented_image = np.reshape(Segemented_image, dims)
    Segemented_image = np.round(Segemented_image)
    # print(data)
    return Segemented_image, classified, variance

path = 'images'
file = 'preserved-london-general-routemaster-bus-number-rml-899-at-north-weald-in-essex-py493b.jpg'
fullfile = os.path.join(path,file)
I = cv2.imread(fullfile)


k  = 3
# data = np.sort(rng.choice(10000, size=(10,10,3), replace=False))
data = I
Segemented_image, classified, variance= k_means_3d(data,k)
print(classified)
data_segemented = np.reshape(classified, data.shape[:-1])
print(data_segemented)


print(data_segemented.shape)
print(I.shape)
cv2.waitKey(0)
Segemented_image = Segemented_image.astype('uint8')
Hori = np.concatenate((I, Segemented_image), axis=1)
cv2.imshow('Hori',Hori)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_org.png',I)
cv2.imwrite('image_compressed.png',Segemented_image)
# plt.imshow(Segemented_image)
# plt.show()