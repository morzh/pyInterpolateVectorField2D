import numpy as np
import  myFlow
import matplotlib.pyplot as plt

      
lms_src = np.random.rand(4,2)
lms_dst =np.random.rand(4,2)

plt.scatter(lms_src[:,0], lms_src[:,1], s=30, c='k')
plt.scatter(lms_src[:,0], lms_src[:,1], s=10, c='r')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=30, c='k')
plt.scatter(lms_dst[:,0], lms_dst[:,1], s=10, c='c')
plt.show()

flow = myFlow()

flow.