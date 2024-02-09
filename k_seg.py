import numpy as np

# def k_means(data, k):
k =3
NumOfPoints = 12
rng = np.random.default_rng()

# data = np.random.rand(1,NumOfPoints)

data = np.sort(rng.choice(100, size=NumOfPoints, replace=False))
data = np.array([1,3,6,10,40,45,66,70,102,110,130,200])
data.astype(float)
L = data.shape[0]
means = np.sort(rng.choice(L, size=k, replace=False))
means = np.array([ data[idx] for idx in means])
means = np.float32(means)

data2 = np.reshape(data, (NumOfPoints,1))
classified = np.argmin( abs(data2 - means), axis=1)
print(classified)
for idx in range(k):
    means[idx] = np.mean(data[classified==idx])
print(means)
    # return result