import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
import numpy as np
import os
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import _CNN
from data_util import *
import random
from PIL import Image




# This is a color map that you can use to plot the SHAP heatmap on the input MRI
colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


# Returns two data loaders (objects of the class: torch.utils.data.DataLoader) that are
# used to load the background and test datasets.
def prepare_dataloaders(bg_csv, test_csv, bg_batch_size = 8, test_batch_size= 1, num_workers=1):
    '''
    Attributes:
        bg_csv (str): The path to the background CSV file.
        test_csv (str): The path to the test data CSV file.
        bg_batch_size (int): The batch size of the background data loader
        test_batch_size (int): The batch size of the test data loader
        num_workers (int): The number of sub-processes to use for dataloader
    '''
    # YOUR CODE HERE
    test_data=CNN_Data(test_csv)
    bg_data=CNN_Data(bg_csv)
    test_dataloader = DataLoader(test_data, test_batch_size, num_workers)
    bg_dataloader = DataLoader(bg_data, bg_batch_size, num_workers)
    return test_dataloader, bg_dataloader
    pass

def bg_prepare_dataloaders(bg_csv, bg_batch_size = 8, shuffle=False, num_workers=1):
    '''
    Attributes:
        bg_csv (str): The path to the background CSV file.
        bg_batch_size (int): The batch size of the background data loader
    '''
    # YOUR CODE HERE
    
    bg_data=CNN_Data(bg_csv)
    bg_dataloader = DataLoader(bg_data, bg_batch_size,num_workers=1)
    return bg_dataloader
    pass

def test_prepare_dataloaders(test_csv, test_batch_size= 1,shuffle=False, num_workers=1):
    '''
    Attributes:
        test_csv (str): The path to the test data CSV file.
        test_batch_size (int): The batch size of the test data loader
    '''
    # YOUR CODE HERE
    test_data=CNN_Data(test_csv)
    test_dataloader = DataLoader(test_data, test_batch_size, num_workers=1)
    return test_dataloader
    pass

# Generates SHAP values for all pixels in the MRIs given by the test_loader
def create_SHAP_values(bg_loader, test_loader, mri_count, save_path):
    '''
    Attributes:
        bg_loader (torch.utils.data.DataLoader): Dataloader instance for the background dataset.
        test_loader (torch.utils.data.DataLoader): Dataloader instance for the test dataset.
        mri_count (int): The total number of explanations to generate.
        save_path (str): The path to save the generated SHAP values (as .npy files).
    '''
    # YOUR CODE HERE
    bg_batch=next(iter(bg_loader))
    bg_data, bg_filename, bg_label=bg_batch
    test=iter(test_loader)
    for i in range(mri_count):
        test_batch=next(test)
        test_data, test_filename, test_label=test_batch
        explain=shap.DeepExplainer(CNN_model, bg_data)
        shap_val=explain.shap_values(test_data)
        np.save(save_path+test_filename[0], np.array(shap_val))
    
    pass

# Aggregates SHAP values per brain region and returns a dictionary that maps 
# each region to the average SHAP value of its pixels. 
def aggregate_SHAP_values_per_region(shap_values_path, seg_path, brain_regions):
    '''
    Attributes:
        shap_values (ndarray): The shap values for an MRI (.npy).
        seg_path (str): The path to the segmented MRI (.nii). 
        brain_regions (dict): The regions inside the segmented MRI image (see data_utl.py)
    '''
    # YOUR CODE HERE
    shape_values=np.load(shap_values_path)
    seg=nib.load(seg_path)
    seg=seg.get_fdata()

    region_score={}
    for region in brain_regions:
        seg_region=seg.copy()
        seg_region[seg_region!=region]=0
        seg_region[seg_region==region]=1
        region_score[region]=np.sum(shape_values*seg_region)/np.count_nonzero(seg_region)
    return region_score
    pass

def cal_avg_SHAP(MRI_list):
    score_dicts=[]
    for i in range(len(MRI_list)):
        filename=MRI_list[i]

        #call score dict per mri file
        shap_values_path= args.outputFolder+"/SHAP/data/"+filename
        str=args.dataFolder+"/seg/"+filename
        seg_path=str[:-4]+'.nii'
        score_dicts.append(aggregate_SHAP_values_per_region(shap_values_path, seg_path, brain_regions))

    avg_dict={}
    for i in range(1, len(score_dicts[0])+1): #run region
        score=0
        for j in range(len(score_dicts)): #run mri dict
            score+=score_dicts[j][i]
        avg=score/len(score_dicts)
        avg_dict[i]=avg
        
    return avg_dict
    pass

# Returns a list containing the top-5 most contributing brain regions to each predicted class (AD/NotAD).
def output_top_5_lst(csv_file, region_score):
    '''
    Attribute:
        csv_file (str): The path to a CSV file that contains the aggregated SHAP values per region.
        region_score(dict):
    '''
    # YOUR CODE HERE
    score=[]
    for i in range(1, len(region_score)+1):
        score.append(region_score[i])
    
    sorted_score=sorted(score, reverse=True)   
    top5_score=sorted_score[:5]
    
    top5_key=[]
    for i in range(5):
        key=score.index(top5_score[i])+1
        top5_key.append(key)
    
    region=[]
    for i in range(5):
        region.append(brain_regions[top5_key[i]])

    data={"Region number":top5_key, "region":region, "value":top5_score}
    df=pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    pass

# Plots SHAP values on a 2D slice of the 3D MRI. 
def plot_shap_on_mri(subject_mri, shap_values, heatmap_filename):
    '''
    Attributes:
        subject_mri (str): The path to the MRI (.npy).
        shap_values (str): The path to the SHAP explanation that corresponds to the MRI (.npy).
        heatmap_filename(str): The file name of the output heatmaps.
    '''
    # YOUR CODE HERE
    shap_val=np.load(shap_values) #load from path
    MRI_data=np.load(subject_mri)

    shap_val=np.swapaxes(shap_val, -4, -1)
    MRI_data=MRI_data[np.newaxis, np.newaxis, :, :, :]
    MRI_data=np.swapaxes(MRI_data, -4, -1)

    shap_slice=np.mean(shap_val, 4, keepdims=False)
    MRI_slice=np.mean(MRI_data, 3, keepdims=False)
    shap.image_plot([slice for slice in shap_slice], -MRI_slice, show=False)
    plt.savefig(args.dataFolder+"/"+heatmap_filename+"_1.png")

    shap_slice=np.mean(shap_val, 3, keepdims=False)
    MRI_slice=np.mean(MRI_data, 2, keepdims=False)
    shap.image_plot([slice for slice in shap_slice], -MRI_slice, show=False)
    plt.savefig(args.dataFolder+"/"+heatmap_filename+"_2.png")

    shap_slice=np.mean(shap_val, 2, keepdims=False)
    MRI_slice=np.mean(MRI_data, 1, keepdims=False)
    shap.image_plot([slice for slice in shap_slice], -MRI_slice, show=False)
    plt.savefig(args.dataFolder+"/"+heatmap_filename+"_3.png")

    img1 = Image.open(args.dataFolder+"/"+heatmap_filename+"_1.png")
    img2 = Image.open(args.dataFolder+"/"+heatmap_filename+"_2.png")
    img3 = Image.open(args.dataFolder+"/"+heatmap_filename+"_3.png")
    fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 1, figsize=(5, 10))
    ax_1.imshow(img1)
    ax_1.axis("off")
    ax_2.imshow(img2)
    ax_2.axis("off")
    ax_3.imshow(img3)
    ax_3.axis("off")
    fig.savefig("./ADNI3/output/heatmaps/"+heatmap_filename+".png")
    pass


if __name__ == '__main__':

    parser=argparse.ArgumentParser() 
    parser.add_argument("--task",type=int,help='The number of task',default=1)
    parser.add_argument("--dataFolder",type=str,help='The path of friends.txt',default="./ADNI3")
    parser.add_argument("--outputFolder",type=str,help='The path of ratings.txt',default="./ADNI3/output")
    args=parser.parse_args()

    PATH=args.dataFolder+"/cnn_best.pth"
    split_csv(args.dataFolder+"/ADNI3.csv")
    CNN_model = _CNN(fil_num=20, drop_rate=0.5)
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    CNN_model.load_state_dict(checkpoint["state_dict"])
    CNN_model.eval()
    test_dataloader=test_prepare_dataloaders(args.dataFolder+"/Test data.csv")
    bg_dataloader=test_prepare_dataloaders(args.dataFolder+"/Background data.csv")


    # TASK I: Load CNN model and isntances (MRIs)
    #         Report how many of the 19 MRIs are classified correctly
    # YOUR CODE HERE 
    if args.task==1:
        correct=0
        incorrect=0
    
        for idx, (MRI_data, filename, label) in enumerate(test_dataloader, 1):
            predict=CNN_model(MRI_data)
            predict=predict.max(dim=1, keepdim=False)[1]
            if predict.sum().item()==label:
                correct+=1    
            else:
                incorrect+=1
                    
        for idx, (MRI_data, filename, label) in enumerate(bg_dataloader, 1):
            predict=CNN_model(MRI_data)
            predict=predict.max(dim=1, keepdim=False)[1]
            if predict.sum().item()==label:
                correct+=1
            else:
                incorrect+=1
            
        data={"Classified": ["Correct", "Incorrect"], "Value": [correct, incorrect]}
        df=pd.DataFrame(data)
        df.to_csv(args.outputFolder+"/task-1.csv", index=False)

    # TASK II: Probe the CNN model to generate predictions and compute the SHAP 
    #          values for each MRI using the DeepExplainer or the GradientExplainer. 
    #          Save the generated SHAP values that correspond to instances with a
    #          correct prediction into output/SHAP/data/
    # YOUR CODE HERE 
    if args.task==2:
        create_SHAP_values(bg_dataloader, test_dataloader, 5, args.outputFolder+"/SHAP/data/")
    



    # TASK III: Plot an explanation (pixel-based SHAP heatmaps) for a random MRI. 
    #           Save heatmaps into output/SHAP/heatmaps/
    # YOUR CODE HERE 

    #select mri randomly
    if args.task==3:
        filename, label=read_csv(args.dataFolder+"/Test data.csv")

        AD_MRI=[]
        NOT_AD_MRI=[]
        for i in range(len(label)):
            if label[i]==1:
                AD_MRI.append(filename[i])
            else:
                NOT_AD_MRI.append(filename[i])

        def random_mri(mri_list): 
            random.seed(6)    
            idx = random.randint(0, len(mri_list)-1)
            MRI_filename=mri_list[idx]
            return MRI_filename

        AD_filename=random_mri(AD_MRI)
        NOT_AD_filename=random_mri(NOT_AD_MRI)
        
        #plot 
        plot_shap_on_mri(args.dataFolder+"/"+AD_filename, args.outputFolder+"/SHAP/data/"+AD_filename, "AD")
        plot_shap_on_mri(args.dataFolder+"/"+NOT_AD_filename, args.outputFolder+"/SHAP/data/"+NOT_AD_filename, "NOT_AD")

    # TASK IV: Map each SHAP value to its brain region and aggregate SHAP values per region.
    #          Report the top-10 most contributing regions per class (AD/NC) as top10_{class}.csv
    #          Save CSV files into output/top10/
    # YOUR CODE HERE 

    #AD
    if args.task==4:
        filename, label=read_csv(args.dataFolder+"/Test data.csv")

        AD_MRI=[]
        NOT_AD_MRI=[]
        for i in range(len(label)):
            if label[i]==1:
                AD_MRI.append(filename[i])
            else:
                NOT_AD_MRI.append(filename[i])

        def random_mri(mri_list): 
            random.seed(6)    
            idx = random.randint(0, len(mri_list)-1)
            MRI_filename=mri_list[idx]
            return MRI_filename

        AD_filename=random_mri(AD_MRI)
        NOT_AD_filename=random_mri(NOT_AD_MRI)

        AD_region_score=cal_avg_SHAP(AD_MRI)
        output_top_5_lst(args.outputFolder+"/top5/task-4-true.csv", AD_region_score)

        NOT_AD_region_score=cal_avg_SHAP(NOT_AD_MRI)
        output_top_5_lst(args.outputFolder+"/top5/task-4-false.csv", NOT_AD_region_score)
    
        pass


