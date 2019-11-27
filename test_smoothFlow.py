import numpy as np
import myFlow
import matplotlib.pyplot as plt

      
lms_src = np.random.rand(4,2)
lms_dst =np.random.rand(4,2)

vecs = lms_dst - lms_src

plt.scatter(lms_src[:,0], lms_src[:,1], s=30, c='k')
plt.scatter(lms_src[:,0], lms_src[:,1], s=10, c='r')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=30, c='k')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=10, c='c')
plt.quiver(lms_src[:,0], lms_src[:,1], vecs[:,0], vecs[:,1], scale=1.0, units='xy')
plt.axis('equal')
plt.show()



flow = myFlow.InterpolateFlow()
# flow.print()
flow.calc_kernelLandmarks(lms_src.astype(float), lms_dst.astype(float), 1e-4, 3.5)
