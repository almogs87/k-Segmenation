import numpy as np


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

    # L = data.shape[0]
    data_org = data
    dims = data.shape
    depth = dims[-1]
    L = round(np.linalg.det(np.identity(len(dims[:-1])) * dims[:-1]))
    data = np.reshape(data, (L,dims[-1]))
    means = np.sort(rng.choice(L, size=k, replace=False))
    means = np.array([ data[idx,:] for idx in means])
    means = np.float32(means)
    variance = np.zeros(k)
    data2 = np.reshape(data, (L,depth,1))
    diff_norm = np.linalg.norm(data2 - means.T, axis=1)
    classified = np.argmin( diff_norm, axis=1)
    count = 0
    flag = 0
    means_old = 0 + means
    while ( count<max_iter and flag==0) :
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

    # print(data)
    return classified, variance

k  = 3
data = np.sort(rng.choice(10000, size=(10,10,3), replace=False))
classified, variance= k_means_3d(data,k)
print(classified)
data_segemented = np.reshape(classified, data.shape[:-1])
print(data_segemented)