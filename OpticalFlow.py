import numpy as np 
import math
from matplotlib import pyplot as plt

class OpticalFlow:

    def __init__(self):
        
        self.lmbda = 0.001
        self.sigma  = 3.5

        self.pi  =   3.14159265
        self.max_axis_num_divisions = 70

        self.G1 = None
        self.V1 = None
        self.beta = None

        self.mesh_cell_size = 1

        self.grid_x_divs = 1
        self.grid_y_divs = 1


        self.cell_coords = np.empty([0,2])


    def create_regularGrid(self, img_face, plot_grid ):

        max_axis_len = max( img_face.shape[0],  img_face.shape[1])
        max_axis_idx = np.argmax( [img_face.shape[0], img_face.shape[1]] )

        print( 'max_axis_len: ', max_axis_len, 'max_axis_idx: ', max_axis_idx)
        print( img_face.shape[max_axis_idx], ' ', img_face.shape[1-max_axis_idx])

        num_max_divisions  = min(max_axis_len, self.max_axis_num_divisions)
        # ratio =  float(img_face.shape[1-max_axis_idx]) /  float(img_face.shape[max_axis_idx])
        # num_sec_divisions = int(ratio*num_max_divisions)
        

        self.mesh_cell_size =  int(( float(img_face.shape[max_axis_idx] ) / float(num_max_divisions)))

        num_max_divisions = math.floor(float(img_face.shape[max_axis_idx] )   / float(self.mesh_cell_size))  + 1.0
        num_sec_divisions = math.floor(float(img_face.shape[1-max_axis_idx] ) / float(self.mesh_cell_size))  + 1.0

        print( 'num_max_divisions: ', num_max_divisions, ' num_sec_divisions: ', num_sec_divisions, '; mesh cell size: ' , self.mesh_cell_size)

        if max_axis_idx == 0:
            self.grid_x_divs = int(num_sec_divisions)
            self.grid_y_divs = int(num_max_divisions)
        elif max_axis_idx == 1:
            self.grid_x_divs = int(num_sec_divisions)
            self.grid_y_divs = int(num_max_divisions)

        grid = np.empty([0,2])

        for y in range(self.grid_y_divs):
            for x in range( self.grid_x_divs ):

                x_coord = x*self.mesh_cell_size
                y_coord = y*self.mesh_cell_size

                node = np.array([x_coord, y_coord])
                grid = np.vstack( (grid, np.round(node)))
        

        if  plot_grid:
            plt.scatter( grid[:,0], grid[:,1], s=1, c='b')
            plt.show()

        return grid.astype('int')


    def build_defomGrid(self, grid, lms):

        grid_vel = np.empty([0,2])


        for idx in range(grid.shape[0]):

            r =  grid[idx, :]
            v_r = self.denseopticalflow (r, lms )
            grid_vel = np.vstack( [grid_vel, v_r])

        return grid_vel


    def calc_cellBarycentricCoords(self, print_info=False, plot_coords=False):

        self.cell_coords = np.empty([0,4])

        num_pts = self.mesh_cell_size

        for idx_x in range(num_pts):
            for idx_y in range(num_pts):

                xx_normalized = float(idx_x)/float(num_pts) 
                yy_normalized = float(idx_y)/float(num_pts) 

                xx = np.array([  xx_normalized, 1-xx_normalized]).reshape((2,1))
                yy = np.array([1-yy_normalized,   yy_normalized]).reshape((1,2))

                bc_coords =  np.matmul(xx, yy).reshape((1,4)).ravel()
                self.cell_coords = np.vstack((self.cell_coords, bc_coords))


        if print_info:
            print('mesh cell size', self.mesh_cell_size)
            # print self.cell_coords

        if plot_coords:
            corners = np.array([[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]])
            points = np.matmul( self.cell_coords, corners)

            plt.scatter( corners[:,0], corners[:,1], c='r', s=35)
            plt.scatter( points[:,0], points[:,1], c='k', s=8)
            plt.show()  
        
    def calc_displacementMaps(self, img_face, grid, grid_deformed, plot_debug):
        #corners = np.array([[0,0], [1,0], [0,1], [1,1]])
        displX = np.zeros( [img_face.shape[0], img_face.shape[1], 2] )
        displY = np.array( [ ] )

        for idx in range(self.grid_x_divs*(self.grid_y_divs-1)):

            if (idx+1) % self.grid_x_divs == 0:
                # print 'skipping loop iteration'
                continue

            corner_00 = grid[idx,                    : ]
            corner_10 = grid[idx+1,                  : ]
            corner_01 = grid[idx+self.grid_x_divs,   : ] 
            corner_11 = grid[idx+self.grid_x_divs+1, : ] 

            corner_deformed_00 = grid_deformed[idx,                    : ]
            corner_deformed_10 = grid_deformed[idx+1,                  : ]
            corner_deformed_01 = grid_deformed[idx+self.grid_x_divs,   : ] 
            corner_deformed_11 = grid_deformed[idx+self.grid_x_divs+1, : ] 

            # print corner_00, ' ', corner_01, ' ', corner_10, ' ', corner_11
            # corners = np.array([[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]])
            '''
            corners          = np.vstack( (corner_00, corner_10, corner_01, corner_11))
            points           = np.matmul( self.cell_coords, corners)
            '''
            corners_deformed = np.vstack( ( corner_deformed_00, corner_deformed_10, corner_deformed_01, corner_deformed_11))
            points_deformed  = np.matmul( self.cell_coords, corners_deformed)


            if plot_debug:
                plt.title( 'self.grid_x_divs: '  + str(self.grid_x_divs) +';  idx is: '+str(idx))
                plt.scatter( corners_deformed[:,0], corners_deformed[:,1], c='r', s=35)
                # plt.scatter( points[:,0],  points[:,1],  c='grey', s=8)
                plt.scatter( points_deformed[:,0],  points_deformed[:,1],  c='k', s=8)
                plt.show()  

        return displX, displY


    def plot_vectorField(self, lms, lms_result):

        lms_vel = lms_result - lms
        plt.quiver( lms[:,0], lms[:,1], lms_vel[:,0], lms_vel[:,1],  linewidth=0.001, scale_units='xy', scale=1.0)
        plt.scatter(lms[:,0], lms[:,1], c='grey', s=1 )
        plt.scatter(lms_result[:,0], lms_result[:,1], c='c', s=1 )
        plt.show()

    def plot_vectorField2(self, lms, lms_vel):
        
        plt.quiver( lms[:,0], lms[:,1], lms_vel[:,0], lms_vel[:,1],   linewidths=0.001, scale=0.01)
        plt.show()

    def plot_vectorField3(self, lms, lms_vel, grid, grid_vel):
        
        scale = 0.01

        plt.quiver( lms[:,0], lms[:,1], scale*lms_vel[:,0], scale*lms_vel[:,1],   linewidths=0.0001, color='g')
        plt.quiver( grid[:,0], grid[:,1], scale*grid_vel[:,0], scale*grid_vel[:,1],   linewidths=0.0001,  color='b')
        plt.show()


    def calc_opticalFlowData(self, lms_src, lms_dst, sigma=3.5, lmbda=1e-3):

        self.sigma = sigma
        self.lmbda = lmbda

        
        num_centroids   =   lms_src.shape[0]
        lms_vel         = lms_dst-lms_src
        G                   =   np.zeros([num_centroids, num_centroids])
        kronecker           =   np.zeros([num_centroids, num_centroids])

        for i  in range(num_centroids):
            for j in range(num_centroids):
                
                # c_i = np.array([ lms_src[i,0], lms_src[i,1] ])
                # c_j = np.array([ lms_src[j,0], lms_src[j,1] ])
                c_i = lms_src[i,:]
                c_j = lms_src[j,:]

                G[i, j] = 1.0/(2*self.pi*self.sigma**2)   *   np.exp((-np.linalg.norm( c_i-c_j )) / (2*self.sigma**2))
                
                if i == j:
                    kronecker[i,j] = 1.0

        self.G1 = G + self.lmbda*kronecker
        self.V1 = lms_vel
        self.beta = np.dot(np.linalg.inv(self.G1), self.V1)

        print('Py version')
        print(self.G1)
        print(self.V1)
        print(self.beta)


    def  denseopticalflow (self, r, centroids):

        v_r = np.zeros([1,2])
        
        for i in range(len(self.beta)):
            
            c_i  = centroids[i, :]
            v_r += self.beta[i, :] / (2*self.pi*self.sigma**2) * np.exp( -1.0*np.linalg.norm( r - c_i ) / (2*self.sigma**2) )  

        return  v_r


