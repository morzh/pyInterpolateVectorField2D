import numpy as np
import myFlow
import matplotlib.pyplot as plt

      
lms_src = np.random.rand(4,2)
lms_dst =np.random.rand(4,2)

plt.scatter(lms_src[:,0], lms_src[:,1], s=30, c='k')
plt.scatter(lms_src[:,0], lms_src[:,1], s=10, c='r')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=30, c='k')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=10, c='c')
plt.show()


X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q, X=0.3, Y=1.1, U=10,  label='Quiver key, length = 10', labelpos='E')

plt.show()


flow = myFlow.InterpolateFlow()
# flow.print()
flow.calc_kernelLandmarks(lms_src.astype(float), lms_dst.astype(float), 1e-4, 3.5)
