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

k =3
NumOfPoints = 12
rng = np.random.default_rng()

data = np.sort(rng.choice(100, size=NumOfPoints, replace=False))
data = np.array([1,3,6,10,40,45,66,70,102,110,130,200])
for tries in range(5):
    classified, variance = k_means(data,3)
    print(np.sum(variance))
    print(str(classified) +'\n')
