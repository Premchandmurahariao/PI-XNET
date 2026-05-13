# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:04:33 2026

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import LineString
from itertools import combinations
import flopy
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from matplotlib.ticker import ScalarFormatter
import gstools as gs
import os
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, r2_score, jaccard_score
#%% Random Generation of Hydraulic conductivity matrices with different complex levels
# Domain dimensions (m)
length, width = 100, 100
cell_size = 1  # m
nx, ny = length // cell_size, width // cell_size  # Number of grid cells

# Number of sets and fractures per set
num_sets = 15000
low_fract_set = 3
medium_fract_set = 6
high_fract_set = 8

def generate_fracture():
    """ Random fracture that starts and ends on different boundaries."""
    global x_values,y_values,start_x ,start_y,end_x ,end_y,start_side,end_side
    start_side = random.choice(["left", "right", "top", "bottom"])
    if start_side == "left":
        start_x = 0
        start_y = random.choice((range(0, width + cell_size, cell_size)))
        end_side = random.choice(["right", "top", "bottom"])
        if end_side =="right":
            end_x = length
            end_y = random.choice((range(0, width + cell_size, cell_size)))
        if end_side =="top":
            end_x = random.choice((range(0, width + cell_size, cell_size)))
            end_y = width
        else:
            end_x = random.choice((range(0, width + cell_size, cell_size)))
            end_y = 0
    if start_side == "right":
        start_x = length
        start_y = random.choice((range(0, width + cell_size, cell_size)))
        end_side = random.choice(["left", "top", "bottom"])
        if end_side =="left":
            end_x = 0
            end_y = random.choice((range(0, width + cell_size, cell_size)))
        if end_side =="top":
            end_x = random.choice((range(0, width + cell_size, cell_size)))
            end_y = width
        else:
            end_x = random.choice((range(0, width + cell_size, cell_size)))
            end_y = 0
    if start_side == "top":
        start_y = width
        start_x = random.choice((range(0, width + cell_size, cell_size)))
        end_side = random.choice(["left", "right", "bottom"])
        if end_side =="left":
            end_x = 0
            end_y = random.choice((range(0, width + cell_size, cell_size)))
        if end_side =="bottom":
            end_x = random.choice((range(0, width + cell_size, cell_size)))
            end_y = 0
        else:
            end_x = length
            end_y = random.choice((range(0, width + cell_size, cell_size)))
    if start_side == "bottom":
        start_y = 0
        start_x = random.choice((range(0, width + cell_size, cell_size)))
        end_side = random.choice(["left", "right", "top"])
        if end_side =="left":
            end_x = 0
            end_y = random.choice((range(0, width + cell_size, cell_size)))
        if end_side =="right":
            end_x = length
            end_y = random.choice((range(0, width + cell_size, cell_size)))
        else:
            end_x = random.choice((range(0, width + cell_size, cell_size)))
            end_y = width
    if (start_x-end_x)> (start_y-end_y):
      x_values = np.linspace(start_x, end_x+cell_size, num=int(abs(start_x - end_x) // cell_size + 1))
      y_values = np.linspace(start_y, end_y+cell_size, num=len(x_values))
    if (start_x-end_x)<(start_y-end_y):
      y_values = np.linspace(start_y, end_y+cell_size, num=int(abs(start_y - end_y) // cell_size + 1))
      x_values = np.linspace(start_x, end_x+cell_size, num=len(y_values))
    return list([start_x,start_y,end_x,end_y])
def generate_fracture_set(num_fractures):
    fractures = []
    lines=[]
    while len(fractures) < num_fractures:
        fracture = generate_fracture()
        fractures.append(fracture)
        lines.append(LineString([(fracture[0],fracture[1]),(fracture[2],fracture[3])]))
    return lines
n = nx * ny
dx = dy = 1.0
# Target real-space mean for ln(K)
mean_K = (-10.30,-13.81)
point_location = random.choice(range(0,num_sets))
low_fracture_sets = []
low_intersection=[];low_areal_density=[];hk_low=np.ones((num_sets,nx, ny));kf=0.01;
hk_low_binary=np.zeros((num_sets,nx, ny));hk_medium_binary=np.zeros((num_sets,nx, ny));hk_high_binary=np.zeros((num_sets,nx, ny))
def sample_variogram_params():
    mean=np.random.uniform(*mean_K)
    return mean
for _ in range(num_sets):
     low_x_cordinates=[];low_y_cordinates=[];
     low_set = generate_fracture_set(low_fract_set)
     low_fracture_sets.append(low_set)
     intersection_points = set()  # Use a set to avoid duplicate points
     fract_length_low=[];
     for liness in low_set:
      frac_length = round(liness.length,0)
      fract_length_low.append(round(liness.length,0))
      interval = cell_size
      points = [liness.interpolate(dist) for dist in range(0, int(frac_length) + 1, interval)]
      low_x_cordinates.append([point.x for point in points])
      low_y_cordinates.append([point.y for point in points])
     low_areal_density.append(np.sum(fract_length_low)/(length*width))
     for line1, line2 in combinations(low_set, 2):
      intersection = line1.intersection(line2)
      if intersection.geom_type == "Point":
        intersection_points.add((intersection.x, intersection.y))
      elif intersection.geom_type == "MultiPoint":
        for point in intersection.geoms:
          intersection_points.add((point.x, point.y))
     if len(list(intersection_points))<=15:
      low_intersection.extend([intersection_points])
     if len(list(intersection_points))>15:
      low_intersection.extend([random.sample(list(intersection_points),15)])
     x = np.arange(nx) * cell_size
     y = np.arange(ny) * cell_size
     mean = sample_variogram_params()
     model = gs.Gaussian(dim=2, var=3, len_scale=[5,5])
     srf = gs.SRF(model, mean=mean)
     logK_field = srf.structured([x, y])
     K_field = np.exp(logK_field)
     hk_low[_,:,:]=K_field 
     for fracture_lst in range(0,low_fract_set):
       for cords in range(0,len(low_x_cordinates[fracture_lst])):
         if 0<= int((low_y_cordinates[fracture_lst][cords]/cell_size)) < ny and 0<=int((low_x_cordinates[fracture_lst][cords]/cell_size))< nx:
           hk_low[_,int((low_y_cordinates[fracture_lst][cords]/cell_size)),int((low_x_cordinates[fracture_lst][cords]/cell_size))]=kf
           hk_low_binary[_,int((low_y_cordinates[fracture_lst][cords]/cell_size)),int((low_x_cordinates[fracture_lst][cords]/cell_size))]=float(1)
           
medium_fracture_sets = []
medium_intersection=[];medium_areal_density=[];hk_medium=np.ones((num_sets,nx, ny));
for _ in range(num_sets):
     medium_x_cordinates=[];medium_y_cordinates=[]
     medium_set = generate_fracture_set(medium_fract_set)
     medium_fracture_sets.append(medium_set)
     intersection_points = set()  # Use a set to avoid duplicate points
     fract_length_medium=[];
     for liness in medium_set:
      frac_length = round(liness.length,0)
      fract_length_medium.append(round(liness.length,0))
      interval = cell_size
      points = [liness.interpolate(dist) for dist in range(0, int(frac_length) + 1, interval)]
      medium_x_cordinates.append([point.x for point in points])
      medium_y_cordinates.append([point.y for point in points])
     medium_areal_density.append(np.sum(fract_length_medium)/(length*width))
     for line1, line2 in combinations(medium_set, 2):  # All unique pairs
      intersection = line1.intersection(line2)
      if intersection.geom_type == "Point":
        intersection_points.add((intersection.x, intersection.y))
      elif intersection.geom_type == "MultiPoint":  # In case of multiple points
        for point in intersection.geoms:
          intersection_points.add((point.x, point.y))
     if len(list(intersection_points))<=15:
      medium_intersection.extend([intersection_points])
     if len(list(intersection_points))>15:
      medium_intersection.extend([random.sample(list(intersection_points),15)])
     x = np.arange(nx) * cell_size
     y = np.arange(ny) * cell_size
     mean = sample_variogram_params()
     model = gs.Gaussian(dim=2, var=3, len_scale=[5,5])
     srf = gs.SRF(model, mean=mean)
     logK_field = srf.structured([x, y])
     K_field = np.exp(logK_field)
     hk_medium[_,:,:]=K_field 
     for fracture_lst in range(0,medium_fract_set):
       for cords in range(0,len(medium_x_cordinates[fracture_lst])):
         if 0<= int((medium_y_cordinates[fracture_lst][cords]/cell_size)) < ny and 0<=int((medium_x_cordinates[fracture_lst][cords]/cell_size))< nx:
           hk_medium[_,int((medium_y_cordinates[fracture_lst][cords]/cell_size)),int((medium_x_cordinates[fracture_lst][cords]/cell_size))]=kf
           hk_medium_binary[_,int((medium_y_cordinates[fracture_lst][cords]/cell_size)),int((medium_x_cordinates[fracture_lst][cords]/cell_size))]=float(1)
high_fracture_sets = []
high_intersection=[]; high_areal_density=[];hk_high=np.ones((num_sets,nx, ny));
for _ in range(num_sets):
     high_x_cordinates=[];high_y_cordinates=[];
     high_set = generate_fracture_set(high_fract_set)
     high_fracture_sets.append(high_set)
     intersection_points = set()
     fract_length_high=[];
     for liness in high_set:
      frac_length = round(liness.length,0)
      fract_length_high.append(round(liness.length,0))
      interval = cell_size
      points = [liness.interpolate(dist) for dist in range(0, int(frac_length) + 1, interval)]
      high_x_cordinates.append([point.x for point in points])
      high_y_cordinates.append([point.y for point in points])
     high_areal_density.append(np.sum(fract_length_high)/(length*width))
     for line1, line2 in combinations(high_set, 2):
      intersection = line1.intersection(line2)
      if intersection.geom_type == "Point":
        intersection_points.add((intersection.x, intersection.y))
      elif intersection.geom_type == "MultiPoint":
        for point in intersection.geoms:
          intersection_points.add((point.x, point.y))
     if len(list(intersection_points))<=15:
      high_intersection.extend([intersection_points])
     if len(list(intersection_points))>15:
      high_intersection.extend([random.sample(list(intersection_points),15)])
     x = np.arange(nx) * cell_size
     y = np.arange(ny) * cell_size
     mean = sample_variogram_params()
     model = gs.Gaussian(dim=2, var=3, len_scale=[5,5])
     srf = gs.SRF(model, mean=mean)
     logK_field = srf.structured([x, y])
     K_field = np.exp(logK_field)
     hk_high[_,:,:]=K_field 
     for fracture_lst in range(0,high_fract_set):
       for cords in range(0,len(high_x_cordinates[fracture_lst])):
         if 0<= int((high_y_cordinates[fracture_lst][cords]/cell_size)) < ny and 0<=int((high_x_cordinates[fracture_lst][cords]/cell_size))< nx:
           hk_high[_,int((high_y_cordinates[fracture_lst][cords]/cell_size)),int((high_x_cordinates[fracture_lst][cords]/cell_size))]=kf
           hk_high_binary[_,int((high_y_cordinates[fracture_lst][cords]/cell_size)),int((high_x_cordinates[fracture_lst][cords]/cell_size))]=float(1)
         
grid_size = 9 
length = 100.0
x_mon = y_mon = np.linspace(10, 90, grid_size)

grid_x, grid_y = np.meshgrid(x_mon, y_mon)
# Steady State Groundwater Model
Monitoring_heads=np.ones(np.shape(grid_x))
nlay, nrow, ncol = 1, 100, 100
delr = delc = 1.0
pumping_points_x_cordinates=[5,15,35,55,95]
pumping_points_y_cordinates=[5,25,75,55,95]
top = 100
botm = 50
Q=-0.001
channels=(len(pumping_points_x_cordinates)+2*len(pumping_points_x_cordinates))
head_matrix_low=np.ones((num_sets,nrow,ncol,channels))
head_matrix_medium=np.ones((num_sets,nrow,ncol,channels))
head_matrix_high=np.ones((num_sets,nrow,ncol,channels))
const_boundary1=100;const_boundary2=100;
os.chdir(r"D:\PhD_folder\IITH\ContaminantHydrology\Codes")
workspace = "./mf6_steady_model"
if not os.path.exists(workspace):
    os.makedirs(workspace)
sim_name = "steady_model"
#model_name="mf6.5.0_linux/bin/mf6" # Activate this cell when you run this on Jupyter notebook
model_name="mf6" # Activate this cell when you run this on Spyder
for complex_frac in range(0,3):
    if complex_frac ==0:
        K = hk_low
    if complex_frac ==1:
        K = hk_medium
    if complex_frac ==2:
        K= hk_high  
    for realizations in range(0,num_sets):
        initial=0
        for pumping_tests in range(0,int(len(pumping_points_x_cordinates))):
            sim = flopy.mf6.MFSimulation(sim_name=sim_name,exe_name=model_name,version="mf6",sim_ws=workspace)
            tdis = flopy.mf6.ModflowTdis(sim, time_units="seconds",nper=1, perioddata=[(1.0, 1, 1)])
            flow_model = flopy.mf6.ModflowGwf(sim, modelname=sim_name, save_flows=True)
            ims = flopy.mf6.ModflowIms(sim, print_option="SUMMARY",complexity=("COMPLEX"),linear_acceleration='BICGSTAB',under_relaxation='DBD',under_relaxation_gamma=0.2,under_relaxation_theta=0.7,under_relaxation_kappa=0.1,under_relaxation_momentum=0.001)
            sim.register_ims_package(ims, [flow_model .name])
            dis = flopy.mf6.ModflowGwfdis(flow_model ,nlay=nlay,nrow=nrow,ncol=ncol,delr=delr,delc=delc,top=top,botm=botm,length_units="METERS")
            npf=flopy.mf6.ModflowGwfnpf(flow_model,icelltype=0,k=K[realizations,:,:],xt3doptions="xt3d rhs",save_specific_discharge=True)
            initial_heads=(np.ones((nlay,nrow,ncol)))*75
            start=initial_heads
            ic=flopy.mf6.ModflowGwfic(flow_model ,pname="ic",strt=start)
            chd_rec = []
            bound_rows=[0,nrow-1]
            for m in range(0,nlay):
               for row_col in range(0,nrow):
                   chd_rec.append(((m,row_col,0),const_boundary1))
                   chd_rec.append(((m,row_col,ncol-1),const_boundary2))
            wel_rec=[(nlay-nlay,int(nrow-pumping_points_y_cordinates[pumping_tests]),int(pumping_points_x_cordinates[pumping_tests]),Q)]
            wel = flopy.mf6.ModflowGwfwel(flow_model , stress_period_data=wel_rec)       
            headfile="{}.hds".format(sim_name)
            head_filerecord=[headfile]
            budgetfile="{}.cbb".format(sim_name)
            budget_filerecord=[budgetfile]
            saverecord=[("HEAD","ALL"),("BUDGET","ALL")]
            printrecord=[("HEAD","LAST")]
            oc=flopy.mf6.ModflowGwfoc(flow_model , saverecord=saverecord,head_filerecord=head_filerecord,budget_filerecord=budget_filerecord,printrecord=printrecord)
            sim.write_simulation(silent=True)
            success,buff=sim.run_simulation(silent=True)
            if not success:
                continue
            headfile = os.path.join(workspace, f"{sim_name}.hds")
            hds = flopy.utils.HeadFile(headfile)
            head = hds.get_data(kstpkper=(0, 0))
            
            dinterpolated_heads_dx = (head[0,:, 2:] - head[0,:, :-2]) / (2 * delc) 
            dinterpolated_heads_dy = (head[0,2:, :] - head[0,:-2, :]) / (2 * delr)
            
            dinterpolated_heads_dx = np.pad(dinterpolated_heads_dx, ((0, 0), (1, 1)), mode='edge')  
            dinterpolated_heads_dy = np.pad(dinterpolated_heads_dy, ((1, 1), (0, 0)), mode='edge')
            if complex_frac == 0:
                head_matrix_low[realizations,:,:,initial]=head[0,:,:]
                head_matrix_low[realizations,:,:,initial+1]=dinterpolated_heads_dx
                head_matrix_low[realizations,:,:,initial+2]=dinterpolated_heads_dy
                initial=initial+3
            if complex_frac == 1:
                head_matrix_medium[realizations,:,:,initial]=head[0,:,:]
                head_matrix_medium[realizations,:,:,initial+1]=dinterpolated_heads_dx
                head_matrix_medium[realizations,:,:,initial+2]=dinterpolated_heads_dy
                initial=initial+3
            if complex_frac == 2:
                head_matrix_high[realizations,:,:,initial]=head[0,:,:]
                head_matrix_high[realizations,:,:,initial+1]=dinterpolated_heads_dx
                head_matrix_high[realizations,:,:,initial+2]=dinterpolated_heads_dy
                initial=initial+3
K_matrix=np.concatenate((hk_low,hk_medium,hk_high))
K_matrix=np.log(K_matrix)
Fracture_matrix=np.concatenate((hk_low_binary,hk_medium_binary,hk_high_binary))
H_matrix=np.concatenate((head_matrix_low,head_matrix_medium,head_matrix_high))
np.save("K_matrix.npy",K_matrix)
np.save("Fracture_matrix.npy",Fracture_matrix)
np.save("H_matrix.npy",H_matrix)     
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, jaccard_score
# PI-XNET Algorithm Starts
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, jaccard_score

# --- Constants & Configuration ---
num_sets = 15000
input_channels = 15
batch_size = 96
cell_size = 1.0 
train_percentage = 0.67
validation_percentage = 0.13

class CrossFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, stream_main, stream_cross):
        combined = torch.cat([stream_main, stream_cross], dim=1)
        return self.fusion(combined)

class FeatureFusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # In the diagram, these are labeled "Block of 3 layers"
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_block(x)

class DecoderBlock(nn.Module):
    def __init__(self, concat_channels, out_channels):
        super().__init__()
        self.fusion_layer = FeatureFusionLayer(concat_channels, out_channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.fusion_layer(x)
        x = self.conv_block(x)
        return x

class PI_XNET(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super().__init__()
        # Encoder Streams
        self.enc1_r = EncoderBlock(input_channels, 64); self.enc1_s = EncoderBlock(input_channels, 64)
        self.enc2_r = EncoderBlock(64, 128); self.enc2_s = EncoderBlock(64, 128)
        self.enc3_r = EncoderBlock(128, 256); self.enc3_s = EncoderBlock(128, 256)
        self.enc4_r = EncoderBlock(256, 512); self.enc4_s = EncoderBlock(256, 512)
        
        # Explicit Bottleneck Blocks (Added to match diagram)
        self.bottleneck_r = EncoderBlock(512, 512)
        self.bottleneck_s = EncoderBlock(512, 512)
        
        # Cross-connection layers (1x1 Convolutions)
        self.enc_cross_r = nn.ModuleList([nn.Conv2d(c, c, 1) for c in [64, 128, 256, 512, 512]])
        self.enc_cross_s = nn.ModuleList([nn.Conv2d(c, c, 1) for c in [64, 128, 256, 512, 512]])
        
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        
        # Decoder Streams
        self.dec4_r = DecoderBlock(512 + 512, 512); self.dec4_s = DecoderBlock(512 + 512, 512)
        self.dec3_r = DecoderBlock(256 + 256, 256); self.dec3_s = DecoderBlock(256 + 256, 256)
        self.dec2_r = DecoderBlock(128 + 128, 128); self.dec2_s = DecoderBlock(128 + 128, 128)
        self.dec1_r = DecoderBlock(64 + 64, 64); self.dec1_s = DecoderBlock(64 + 64, 64)
        
        self.trans4_r = nn.Conv2d(512, 256, 1); self.trans4_s = nn.Conv2d(512, 256, 1)
        self.trans3_r = nn.Conv2d(256, 128, 1); self.trans3_s = nn.Conv2d(256, 128, 1)
        self.trans2_r = nn.Conv2d(128, 64, 1); self.trans2_s = nn.Conv2d(128, 64, 1)

        self.dec_cross_r = nn.ModuleList([CrossFusion(c) for c in [512, 256, 128, 64]])
        self.dec_cross_s = nn.ModuleList([CrossFusion(c) for c in [512, 256, 128, 64]])
        
        self.final_seg = nn.Conv2d(64, num_classes, 1)
        self.final_reg = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        f1_r_pre, f1_s_pre = self.enc1_r(x), self.enc1_s(x)
        f1_r = f1_r_pre + self.enc_cross_s[0](f1_s_pre)
        f1_s = f1_s_pre + self.enc_cross_r[0](f1_r_pre)
        sz1 = f1_r.size()
        xr, idx1_r = self.pool(f1_r)
        xs, idx1_s = self.pool(f1_s)
        
        f2_r_pre, f2_s_pre = self.enc2_r(xr), self.enc2_s(xs)
        f2_r = f2_r_pre + self.enc_cross_s[1](f2_s_pre)
        f2_s = f2_s_pre + self.enc_cross_r[1](f2_r_pre)
        sz2 = f2_r.size()
        xr, idx2_r = self.pool(f2_r)
        xs, idx2_s = self.pool(f2_s)
        
        f3_r_pre, f3_s_pre = self.enc3_r(xr), self.enc3_s(xs)
        f3_r = f3_r_pre + self.enc_cross_s[2](f3_s_pre)
        f3_s = f3_s_pre + self.enc_cross_r[2](f3_r_pre)
        sz3 = f3_r.size()
        xr, idx3_r = self.pool(f3_r)
        xs, idx3_s = self.pool(f3_s)

        f4_r_pre, f4_s_pre = self.enc4_r(xr), self.enc4_s(xs)
        f4_r = f4_r_pre + self.enc_cross_s[3](f4_s_pre)
        f4_s = f4_s_pre + self.enc_cross_r[3](f4_r_pre)
        sz4 = f4_r.size()
        xr, idx4_r = self.pool(f4_r)
        xs, idx4_s = self.pool(f4_s)

        bn_r_pre = self.bottleneck_r(xr)
        bn_s_pre = self.bottleneck_s(xs)
        
        xr = bn_r_pre + self.enc_cross_s[4](bn_s_pre)
        xs = bn_s_pre + self.enc_cross_r[4](bn_r_pre)

    
        ur4 = self.unpool(xr, idx4_r, output_size=sz4)
        us4 = self.unpool(xs, idx4_s, output_size=sz4)
        dr4 = self.dec4_r(torch.cat([ur4, f4_r], dim=1))
        ds4 = self.dec4_s(torch.cat([us4, f4_s], dim=1))
        dr4_f, ds4_f = self.dec_cross_r[0](dr4, ds4), self.dec_cross_s[0](ds4, dr4)

        dr4_trans = self.trans4_r(dr4_f)
        ds4_trans = self.trans4_s(ds4_f)
        ur3 = self.unpool(dr4_trans, idx3_r, output_size=sz3)
        us3 = self.unpool(ds4_trans, idx3_s, output_size=sz3)
        dr3 = self.dec3_r(torch.cat([ur3, f3_r], dim=1))
        ds3 = self.dec3_s(torch.cat([us3, f3_s], dim=1))
        dr3_f, ds3_f = self.dec_cross_r[1](dr3, ds3), self.dec_cross_s[1](ds3, dr3)

        dr3_trans = self.trans3_r(dr3_f)
        ds3_trans = self.trans3_s(ds3_f)
        ur2 = self.unpool(dr3_trans, idx2_r, output_size=sz2)
        us2 = self.unpool(ds3_trans, idx2_s, output_size=sz2)
        dr2 = self.dec2_r(torch.cat([ur2, f2_r], dim=1))
        ds2 = self.dec2_s(torch.cat([us2, f2_s], dim=1))
        dr2_f, ds2_f = self.dec_cross_r[2](dr2, ds2), self.dec_cross_s[2](ds2, dr2)

        dr2_trans = self.trans2_r(dr2_f)
        ds2_trans = self.trans2_s(ds2_f)
        ur1 = self.unpool(dr2_trans, idx1_r, output_size=sz1)
        us1 = self.unpool(ds2_trans, idx1_s, output_size=sz1)
        dr1 = self.dec1_r(torch.cat([ur1, f1_r], dim=1))
        ds1 = self.dec1_s(torch.cat([us1, f1_s], dim=1))
        dr1_f, ds1_f = self.dec_cross_r[3](dr1, ds1), self.dec_cross_s[3](ds1, dr1)

        return {
            'regression': self.final_reg(dr1_f), 
            'segmentation': self.final_seg(ds1_f)
        }

grid_size = 9
x_mon = y_mon = np.linspace(10, 90, grid_size)
grid_x, grid_y = np.meshgrid(x_mon, y_mon)
collocation_points = torch.tensor(np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)).round().long()
pump_locations = [(5, 5), (25, 15), (75, 35), (55, 55), (95, 95)]
pump_points = torch.tensor(pump_locations, dtype=torch.long)
all_points = torch.unique(torch.cat([collocation_points, pump_points]), dim=0)

def pde_constraint_loss(pred_k, input_tensor, all_points, scaler_y, scalers_x):
    device = pred_k.device
    mu_logK = torch.tensor(scaler_y.mean_, device=device).view(1,1,1,1)
    std_logK = torch.tensor(scaler_y.scale_, device=device).view(1,1,1,1)

    logK = pred_k * std_logK + mu_logK
    K = torch.exp(logK)
    K = torch.clamp(K, min=1e-7, max=1e7)
    dh_dx_norm = input_tensor[:, 1::3, :, :]
    dh_dy_norm = input_tensor[:, 2::3, :, :]

    std_dhx = torch.tensor(
        [scalers_x[i].scale_[0] for i in range(1, input_tensor.shape[1], 3)],
        device=device
    ).view(1, -1, 1, 1)

    std_dhy = torch.tensor(
        [scalers_x[i].scale_[0] for i in range(2, input_tensor.shape[1], 3)],
        device=device
    ).view(1, -1, 1, 1)

    dh_dx = dh_dx_norm * std_dhx
    dh_dy = dh_dy_norm * std_dhy

    grad_x = dh_dx.mean(dim=1, keepdim=True)
    grad_y = dh_dy.mean(dim=1, keepdim=True)

    grad_x = torch.clamp(grad_x, -1e3, 1e3)
    grad_y = torch.clamp(grad_y, -1e3, 1e3)
    qx = -K * grad_x
    qy = -K * grad_y
    dqdx = (qx[:, :, 2:, 1:-1] - qx[:, :, :-2, 1:-1]) / (2 * cell_size)
    dqdy = (qy[:, :, 1:-1, 2:] - qy[:, :, 1:-1, :-2]) / (2 * cell_size)

    div_q = dqdx + dqdy
    div_q_padded = F.pad(div_q, (1,1,1,1))
    pumping_mask = torch.zeros_like(div_q_padded)
    expected_div = torch.zeros_like(div_q_padded)
    pump_rates = [0.001]*len(pump_locations)

    for (x, y), rate in zip(pump_locations, pump_rates):
        if x < div_q_padded.shape[2] and y < div_q_padded.shape[3]:
            pumping_mask[:, 0, x, y] = 1.0
            expected_div[:, 0, x, y] = rate/(cell_size**2)
            
    idx_x = all_points[:, 0]
    idx_y = all_points[:, 1]

    div_q_colloc = div_q_padded[:, 0, idx_x, idx_y]
    expected_div_colloc = expected_div[:, 0, idx_x, idx_y]
    pumping_mask_colloc = pumping_mask[:, 0, idx_x, idx_y]
    squared_error = (div_q_colloc - expected_div_colloc) ** 2
    weight_mask = 1.0 + 3.0 * pumping_mask_colloc
    weighted_error = squared_error * weight_mask
    return weighted_error.mean()

class UnifiedLoss(nn.Module):    
    def __init__(self, all_points, scaler_y, scalers_x, k_threshold_val=0.01):
        super().__init__()
        self.all_points = all_points
        self.scaler_y = scaler_y
        self.scalers_x = scalers_x
        
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_pde = nn.Parameter(torch.zeros(1))
        self.log_weight_hard = nn.Parameter(torch.tensor(0.0))
        
        norm_thresh = (np.log(k_threshold_val) - scaler_y.mean_[0]) / scaler_y.scale_[0]
        self.register_buffer('k_threshold', torch.tensor(norm_thresh, dtype=torch.float32))

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_k, true_k, pred_seg, true_seg, input_tensor):
        l_seg = self.ce_loss(pred_seg, true_seg.long().squeeze(1))
        l_reg = self.mse_loss(pred_k, true_k)
        l_pde = pde_constraint_loss(pred_k, input_tensor, self.all_points, self.scaler_y, self.scalers_x)
        
        pred_mask = torch.argmax(pred_seg, dim=1, keepdim=True)
        low_k_violations = F.relu(self.k_threshold - pred_k)
        l_hard = (low_k_violations * pred_mask.float()).mean()

        total = (
            torch.exp(-self.log_var_seg) * l_seg + 
            torch.exp(-self.log_var_reg) * l_reg + 
            torch.exp(-self.log_var_pde) * l_pde + 
            torch.exp(self.log_weight_hard) * l_hard +
            self.log_var_seg + self.log_var_reg + self.log_var_pde
        )
        
        return total, {
            'total': total.item(), 
            'seg': l_seg.item(), 
            'reg': l_reg.item(), 
            'pde': l_pde.item(),
            'hard': l_hard.item(),
            'weight_hard': torch.exp(self.log_weight_hard).item()
        }

X = np.load("H_matrix.npy"); Y = np.load("K_matrix.npy"); Y_seg = np.load("Fracture_matrix.npy")
X_reshaped = X.reshape(num_sets*3, 100, 100, input_channels)
X_std = np.zeros_like(X_reshaped); scalers_x = []
for i in range(input_channels):
    s = StandardScaler().fit(X_reshaped[:,:,:,i].reshape(-1,1)); scalers_x.append(s)
    X_std[:,:,:,i] = s.transform(X_reshaped[:,:,:,i].reshape(-1,1)).reshape(num_sets*3,100,100)
scaler_y = StandardScaler().fit(Y.reshape(-1,1))
Y_std = scaler_y.transform(Y.reshape(-1,1)).reshape((num_sets*3),100, 100)
X_t = torch.from_numpy(X_std).permute(0,3,1,2).float(); Y_t = torch.from_numpy(Y_std).unsqueeze(1).float(); Y_seg_t = torch.from_numpy(Y_seg).long()

# Random Sampling 
train_0_100 = np.random.choice(np.arange(0, num_sets), int(train_percentage*num_sets), replace=False)
val_0_100_pool = np.setdiff1d(np.arange(0, num_sets), train_0_100)
val_0_100 = np.random.choice(val_0_100_pool, int(validation_percentage*num_sets), replace=False)

train_100_200 = np.random.choice(np.arange(num_sets, 2*num_sets), int(train_percentage*num_sets), replace=False)
val_100_200_pool = np.setdiff1d(np.arange(num_sets, 2*num_sets), train_100_200)
val_100_200 = np.random.choice(val_100_200_pool, int(validation_percentage*num_sets), replace=False)

train_200_300 = np.random.choice(np.arange(2*num_sets, 3*num_sets), int(train_percentage*num_sets), replace=False)
val_200_300_pool = np.setdiff1d(np.arange(2*num_sets, 3*num_sets), train_200_300)
val_200_300 = np.random.choice(val_200_300_pool, int(validation_percentage*num_sets), replace=False)

train_indices = np.concatenate([train_0_100, train_100_200, train_200_300])
val_indices = np.concatenate([val_0_100, val_100_200, val_200_300])
all_indices = np.arange(X_t.shape[0])
test_indices = np.setdiff1d(all_indices, np.concatenate([train_indices, val_indices]))

np.save('train_indices.npy', train_indices); np.save('val_indices.npy', val_indices)
# Training function
def train_model(model, train_loader, val_loader, num_epochs=45, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = UnifiedLoss(all_points, scaler_y, scalers_x).to(device)
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train(); t_loss = 0
        for xb, yb_reg, yb_seg in train_loader:
            xb, yb_reg, yb_seg = xb.to(device), yb_reg.to(device), yb_seg.to(device)
            optimizer.zero_grad(); out = model(xb)
            loss, _ = loss_fn(out['regression'], yb_reg, out['segmentation'], yb_seg, xb)
            loss.backward(); optimizer.step(); t_loss += loss.item()
        model.eval(); v_loss = 0
        with torch.no_grad():
            for xb, yb_reg, yb_seg in val_loader:
                xb, yb_reg, yb_seg = xb.to(device), yb_reg.to(device), yb_seg.to(device)
                out = model(xb); loss, _ = loss_fn(out['regression'], yb_reg, out['segmentation'], yb_seg, xb); v_loss += loss.item()
        avg_v = v_loss/len(val_loader)
        if avg_v < best_loss: best_loss = avg_v; torch.save(model.state_dict(), 'best_model.pth')
# Testing function
def test_model(model, test_loader, scaler_y, device='cuda', sample_indices=None, save_results=True):
    model.eval(); device = torch.device(device if torch.cuda.is_available() else "cpu")
    num_test = len(test_loader.dataset)
    predicted_K_all = np.zeros((num_test, 100, 100)); true_K_all = np.zeros((num_test, 100, 100))
    metrics = {c: {'segmentation': {'accuracy': [], 'IoU': []}, 'viz_samples': [], 'RMSE': [], 'R²': []} for c in ['low', 'medium', 'high']}

    with torch.no_grad():
        batch_idx = 0
        for xb, yb_reg, yb_seg in test_loader:
            xb, yb_reg, yb_seg = xb.to(device), yb_reg.to(device), yb_seg.to(device)
            outputs = model(xb)
            for i in range(xb.size(0)):
                pk = scaler_y.inverse_transform(outputs['regression'][i].cpu().numpy().reshape(-1, 1)).reshape(100, 100)
                tk = scaler_y.inverse_transform(yb_reg[i].cpu().numpy().reshape(-1, 1)).reshape(100, 100)
                
                curr_idx = batch_idx * test_loader.batch_size + i
                if curr_idx < num_test:
                    predicted_K_all[curr_idx] = pk
                    true_K_all[curr_idx] = tk
                    
                    orig_idx = test_indices[curr_idx]
                    complexity = 'low' if orig_idx < num_sets else ('medium' if orig_idx < 2*num_sets else 'high')
                    
                    pm = torch.argmax(outputs['segmentation'][i], dim=0); tm = yb_seg[i].squeeze()
                    metrics[complexity]['segmentation']['accuracy'].append((pm == tm).sum().item() / 10000 * 100)
                    metrics[complexity]['segmentation']['IoU'].append(jaccard_score(tm.cpu().numpy().ravel(), pm.cpu().numpy().ravel(), average='binary', zero_division=1) * 100)
                    metrics[complexity]['RMSE'].append(np.sqrt(mean_squared_error(tk, pk)))
                    metrics[complexity]['R²'].append(r2_score(tk.ravel(), pk.ravel()))
                    metrics[complexity]['viz_samples'].append((tk, pk, tm.cpu().numpy(), pm.cpu().numpy()))
            batch_idx += 1

    if save_results: np.save('predicted_K_test.npy', predicted_K_all); np.save('true_K_test.npy', true_K_all)
    for c in ['low', 'medium', 'high']:
        if metrics[c]['RMSE']:
            print(f"\n{c.upper()} Metrics: Acc={np.mean(metrics[c]['segmentation']['accuracy']):.2f}, RMSE={np.mean(metrics[c]['RMSE']):.4f}, R2={np.mean(metrics[c]['R²']):.4f}")
        else: print(f"\n{c.upper()} Metrics: No samples in test set.")
    
    if sample_indices:
        for c, idxs in sample_indices.items():
            for idx in idxs:
                avail = len(metrics[c]['viz_samples'])
                if avail > 0:
                    actual_idx = idx if idx >= 0 else avail + idx
                    if 0 <= actual_idx < avail:
                        tk, pk, tm, pm = metrics[c]['viz_samples'][actual_idx]
                        fig, ax = plt.subplots(1, 2); ax[0].imshow(tk); ax[0].set_title(f"True {c}"); ax[1].imshow(pk); ax[1].set_title(f"Pred {c}"); plt.show()
    return metrics

dataset = TensorDataset(X_t, Y_t, Y_seg_t)
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=1, shuffle=False)

model = PI_XNET(input_channels=input_channels)
train_model(model, train_loader, val_loader)
model.load_state_dict(torch.load('best_model.pth'))
test_metrics = test_model(model, test_loader, scaler_y, device='cuda', sample_indices={'low': [-1], 'medium': [-1], 'high': [-1]})
