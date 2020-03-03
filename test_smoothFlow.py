import numpy as np
import myFlow
import matplotlib.pyplot as plt
from OpticalFlow import OpticalFlow

res = 50
lms_src = res*np.random.rand(4,2)
lms_dst = res*np.random.rand(4,2)

vecs = lms_dst - lms_src
'''
plt.scatter(lms_src[:,0], lms_src[:,1], s=30, c='k')
plt.scatter(lms_src[:,0], lms_src[:,1], s=10, c='r')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=30, c='k')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=10, c='c')
plt.quiver(lms_src[:,0], lms_src[:,1], vecs[:,0], vecs[:,1], scale=1.0, units='xy', width=0.06, headwidth=10, color='r')
plt.axis('equal')
plt.show()
'''


c_flow = myFlow.InterpolateFlow()
py_flow = OpticalFlow()


sigma = 3.0
lmbda = 1e-4

c_flow.calc_kernelLandmarks(lms_src, lms_dst, sigma, lmbda)
c_vecfield = c_flow.get_flow((0, 0), (res, res), lms_src)

py_flow.calc_opticalFlowData(lms_src, lms_dst, sigma, lmbda)


X = np.arange(0, res, 1)
Y = np.arange(0, res, 1)
U = c_vecfield[:, :, 0]
V = c_vecfield[:, :, 1]

plt.scatter(lms_src[:,0], lms_src[:,1], s=30, c='k')
plt.scatter(lms_src[:,0], lms_src[:,1], s=10, c='r')
# plt.scatter(lms_dst[:,0], lms_dst[:,1], s=30, c='k')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=10, c='c')
plt.quiver(X, Y, U, V, scale=2.0, units='xy', width=0.006, headwidth=10, color='r')
plt.quiver(lms_src[:,0], lms_src[:,1], vecs[:,0], vecs[:,1], scale=3, units='xy', width=0.06, headwidth=10, color='k')
plt.axis('equal')
plt.tight_layout()
plt.show()
