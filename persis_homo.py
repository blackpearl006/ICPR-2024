'''
This code snippet is can be used to convert the fMR timeseries into PD using ripser and obtain pairwise wasserstein distance across all the ROI's for a given brain network
Input : fMR timeseries
Output : Inter-ROI interactions for a brain network corresponding to Betti Descriptors : H_0, H_1, H_2
Few of the variables are hard-coded here, Change the variables data_dir, subject_classes & class_dir
'''
import os
import gc
import persim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripser import Rips
import warnings
warnings.filterwarnings("ignore")

# networks = ['cerebellum','default_mode','frontoparietal','occipital','cingulo-opercular', 'sensorimotor']
networks = ['CB','DMN','FP','OP','CO','SM']
roi_count = [18,34,21,22,32,33]
descriptors = ['H0','H1','H2']

def get_data(network, subject_class):
    '''
    function to get the data from the csv files
    data is returned as a dictionary object with the keys being the 2 claases 0 (healthy) and 1 (mci) which in turn is a dictionary
    the 2 classes have all subjects as thier key and values are a numpy array of the shape (number of rois, 187)
    Input : folder path to the network folder which has 2 folders healthy and mci which in turn has xx subjects .csv folders
    Output : dictionary[ class { 0, 1} ][ subject_name ][ ROI number ]
    '''
    data_dir = f'/path/to/fMRtimeseries/{subject_class}/{network}'
    data = {}
    for j in os.listdir(data_dir):
        df = pd.read_csv(os.path.join(data_dir,j), header=None)
        if not (df.values == 0).any(axis=1).any():
            data[j] = np.array(df.values)
    return data

def persis_plot(df, tau=1, dim = 3):
    '''This function uses ripser to calculate the Persistance diagram from the 3D point cloud formed using sliding window embedding.
    Input : 1 Dimensional column vector 
    Output : Persistent Diagras corresponding to Dimensions 0, 1 & 2
    dim : set to 3
    tau : time lag set to 1
    '''
    df = df.reshape(-1,1)
    emb_vector = [[ df[i+j*tau][0] for j in range(dim)] for i in range(len(df) - dim*tau)]

    rips = Rips(maxdim=2)
    dgms = rips.fit_transform(np.array(emb_vector))
    H0_dgm = dgms[0]
    H1_dgm = dgms[1]
    H2_dgm = dgms[2]

    return (H0_dgm,H1_dgm,H2_dgm)

subject_classes = ['MCI','CN']
Betti_matrix = {}
for network,rois in zip(networks,roi_count):
    for subject_class in subject_classes:
        Betti_matrix[subject_class]={}
        print('Before')
        data = get_data(network=network,subject_class=subject_class)
        print('After')
        for sub_name in list(data.keys()):
            Betti_matrix[subject_class][sub_name] = {}
            for roi in range(rois):
                Betti_matrix[subject_class][sub_name][roi] = {}
                H0_dgm,H1_dgm,H2_dgm=np.array([]),np.array([]),np.array([])
                
                H0_dgm,H1_dgm,H2_dgm = persis_plot(data[sub_name][roi])
                Betti_matrix[subject_class][sub_name][roi]['H0']=H0_dgm
                Betti_matrix[subject_class][sub_name][roi]['H1']=H1_dgm
                Betti_matrix[subject_class][sub_name][roi]['H2']=H2_dgm


        for sub_name in list(data.keys()):
            class_dir = f'outpath/csv/{network}/{subject_class}/'
            class_dir_fig = f'outpath/img/{network}/{subject_class}/'
            os.makedirs(class_dir,exist_ok=True)
            os.makedirs(class_dir_fig,exist_ok=True)
            dist1, dist2, dist3 = [], [], []
            print(sub_name)
            for roi in range(rois):
                roi_list1,roi_list2,roi_list3=[],[],[]
                roi1_h1 = Betti_matrix[subject_class][sub_name][roi]['H0']
                roi1_h2 = Betti_matrix[subject_class][sub_name][roi]['H1']
                roi1_h3 = Betti_matrix[subject_class][sub_name][roi]['H2']
                for roi2 in range(rois):
                    roi2_h1 = Betti_matrix[subject_class][sub_name][roi2]['H0']
                    roi2_h2 = Betti_matrix[subject_class][sub_name][roi2]['H1']
                    roi2_h3 = Betti_matrix[subject_class][sub_name][roi2]['H2']
                    roi_list1.append(persim.wasserstein(roi1_h1,roi2_h1,matching=False))
                    roi_list2.append(persim.wasserstein(roi1_h2,roi2_h2,matching=False))
                    roi_list3.append(persim.wasserstein(roi1_h3,roi2_h3,matching=False))
                
                dist1.append(roi_list1)
                dist2.append(roi_list2)
                dist3.append(roi_list3)

            pd.DataFrame(dist1).to_csv(f'{class_dir}{sub_name}_h0.csv')
            pd.DataFrame(dist2).to_csv(f'{class_dir}{sub_name}_h1.csv')
            pd.DataFrame(dist3).to_csv(f'{class_dir}{sub_name}_h2.csv')

            for num, dist_img in enumerate([dist1,dist2,dist3]):
                fig = plt.figure(frameon=False)
                fig.set_size_inches(4,4)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                im = ax.imshow(dist_img, aspect='auto', cmap='gray')
                # plt.colorbar(im)
                plt.savefig(f'{class_dir_fig}{sub_name}_h{num}.png',dpi=56, bbox_inches='tight')
                plt.close()
                gc.collect()