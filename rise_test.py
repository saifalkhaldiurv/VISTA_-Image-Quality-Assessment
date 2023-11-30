import socket

import time
# import pandas as pd
import config
from captum.attr._utils.attribution import  Attribution, LayerAttribution
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import torchvision
from pathlib import Path
import torch
import models
from captum import attr
from utils.util import FilteredDatasetFolder,WrappedModel

from interpretability.activation_clustering import plot_activation_clusters, plot_activation_nmf
from interpretability.saliency import plot_activations,plot_average_saliency,plot_saliency_paper
import numpy as np 

from rise import RISE
import matplotlib.pyplot as plt
import torch.nn.functional as F

class PerformanceStatistics():

  def __init__(self,id) -> None:
      self.id=id
      self.times=[]

  def update(self,time):
    self.times.append(time)
  def mean_std(self):
    times = np.array(self.times)
    return times.mean(),times.std()
  
  
  

class AttributionAggregator:
  def __init__(self,name) -> None:
    self.name=name
    self.sums = 0.0
    self.all = []
    self.model = model
  def update(self,value):
    value = value.detach().cpu().numpy().transpose(1,2,0)
    self.sums += value
    self.all.append(value)
  def average(self):
    return self.sums/len(self.all)


#===========================================================================#
def generate_visualizations(model, encoder,dataloader, device,inverse_normalization,id,class_name,k_clustering):
    
    folder = output_folder/id
    folder.mkdir(exist_ok = True,parents=True)

    def wrapped_model(inp):
      # print("old shape", inp.shape)
      # inp = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(inp) #upsample from 640 to 1280
      # inp = F.interpolate(inp, scale_factor=0.5, mode='bilinear', align_corners=False) #downsample from 1280 to 640
      inp = inp.half()
      # print("new shape", inp.shape)
      return model(inp)[1]

    model.eval()
    
    n_samples = len(dataloader)
    i=0
    # TODO this is really ugly, refactor and put computation of attributions in classes, each of which outputs an Aggregator
    aggregators= [AttributionAggregator("Image"),
    AttributionAggregator("Gradient"),
    AttributionAggregator("GradCAM"),
    AttributionAggregator("Occlusion"),
    AttributionAggregator("RISE"),
    ]
    
    statistics = {agg.name:PerformanceStatistics(agg.name) for agg in aggregators}
    
    # remove image_agg
    image_agg = aggregators[0]
    aggregators = aggregators[1:]
    
    rise = RISE(wrapped_model)
    saliency = attr.Saliency(wrapped_model)
    layer_activation = attr.LayerActivation(wrapped_model,encoder)
    gradcam = attr.LayerGradCam(wrapped_model,encoder)
    occlusion = attr.Occlusion(wrapped_model)
    blur_transformation = torchvision.transforms.GaussianBlur(kernel_size=(63,63), sigma=(60, 60))
    for inputs, labels in tqdm(dataloader):
      inputs = inputs.to(device)
      labels = labels.to(device)
      inputs.requires_grad=True

      # Time use of model
      t0 = time.perf_counter()
      wrapped_model(inputs)
      t1 = time.perf_counter()
      statistics["Image"].update(t1-t0)
      
      t0 = time.perf_counter()
      gradient_maps = saliency.attribute(inputs, target=labels,abs=False)
      t1 = time.perf_counter()
      statistics["Gradient"].update(t1-t0)
      t0 = time.perf_counter()
      gradcam_maps = gradcam.attribute(inputs,target=labels)
      t1 = time.perf_counter()
      statistics["GradCAM"].update(t1-t0)
      ## resize gradcam to match input size
      gradcam_maps = LayerAttribution.interpolate(gradcam_maps, inputs.shape[2:4])

      inputs.requires_grad=False
      
      # OCCLUSION
      baselines =  inputs.clone().detach()
      for j in range(baselines.shape[0]):
        baselines[j,]=blur_transformation(baselines[j,])*0.5
      t0 = time.perf_counter()
      occlusion_maps = occlusion.attribute(inputs,target=labels,sliding_window_shapes=(3,240,240),strides=(3,120,120),baselines=baselines,show_progress=False)
      t1 = time.perf_counter()
      statistics["Occlusion"].update(t1-t0)
      
      # RISE
      inputs = (inputs,)
      n_masks = 2048
      initial_mask_shape = (5,5)
      t0 = time.perf_counter()
      importance_maps = rise.attribute(inputs, n_masks=n_masks, initial_mask_shapes=(initial_mask_shape,), target=labels, show_progress=True)
      t1 = time.perf_counter()
      statistics["RISE"].update(t1-t0)
      inputs = inputs[0]

      for j in range(inputs.shape[0]):

        if hasattr(model,"encoder_reshape"):
          # activations=activations.view(*model.encoder_reshape[1:])
          gradcam_maps = gradcam_maps.view(*model.encoder_reshape[1:])
          pass
          
        image, label= inputs[j,],labels[j]

        importance_maps = importance_maps[None, :, :] * np.ones(3, dtype=int)[:, None, None] #torch.Size([1, 3, 640, 640])
        # normalize to [-1,1]
        importance_maps -= importance_maps.min()
        importance_maps /= importance_maps.max()
        # importance_maps *= 2
        # importance_maps -= 1

        maps = [gradient_maps,gradcam_maps,occlusion_maps,importance_maps]
        # torch.save(maps, folder / f'{id}_sample{i}_attribution_tensors.pt')
        with torch.no_grad():
          image = inverse_normalization(image)
          image_agg.update(image)
          image = image_agg.all[-1]
          for agg,value in zip(aggregators,maps):
            agg.update(value[j,])

          saliency_filepath = folder / f"{id}_sample{i}_attribution.png"
          values, names = [agg.all[-1] for agg in aggregators], [agg.name for agg in aggregators]
          plot_saliency_paper(saliency_filepath,image,values,names,class_name)          

          # activations_filepath = folder / f"{id}_sample{i}_activations.png"
          # plot_activations(activations_filepath,image,label,activation,id)
        i += 1
    average_filepath = output_folder/f"{id}_average.png"
    values, names = [agg.average() for agg in aggregators], [agg.name for agg in aggregators]
    plot_saliency_paper(average_filepath,image_agg.average(),values,names,class_name)

    for agg in aggregators:
      all = np.stack([v.sum(axis=2) for v in agg.all],axis=0)
      plot_activation_clusters(all,output_folder/f"{id}_cluster_{agg.name.lower()}.png",k_clustering)
      plot_activation_nmf(all,output_folder/f"{id}_nmf_{agg.name.lower()}.png",k_clustering)
    

    statistics_file = output_folder/f"{socket.gethostname()}_{id}_statistics.txt"
    text=""
    for k,ps in statistics.items():
      mean,std = ps.mean_std()
      n = len(ps.times)
      coefficient_variation = (1+1/(4*n))*(std/mean)
      text += f"{k}:{mean:.4f} ({std:.5f},)\n"
    statistics_file.write_text(text)


# ================================================================ # 



if __name__ == '__main__':

  import warnings
  warnings.filterwarnings("ignore") 

  output_folder = config.base_output_folder/"rise"
  Path(output_folder).mkdir(parents=True,exist_ok=True)


  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model_names = [
    # "FGRNET",
  #  'UNet',
  #  'UNetNoDecoder',
    "vgg16patch"
  ]

  n_samples=4#50
  k_clustering = 2
  for model_name in model_names:
    for class_index,class_name in enumerate(config.class_names):
        
        def filter_sample(fc): 
            # return True
          f,c=fc
          return  c==class_index

        dataset = FilteredDatasetFolder(config.data_dir,filter_sample,config.test_transformation)
        n_class = len(dataset)
        if n_class<n_samples:
          print(f"Class {class_name} has {n_class} samples, less than the required {n_samples}, skipping..")
          continue
        else:
          print(f"Class {class_name} has {n_class} samples, more than the required {n_samples}, computing..")

        # print("Samples: ",image_dataset.samples)
        if not n_samples is None:
          classes = dataset.classes
          dataset = Subset(dataset,list(range(n_samples)))
          dataset.classes = classes
        dataloader= torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3,drop_last=False)
        

        id = f'{model_name.replace("/","_")}_class{class_index}'
        print(f"Generating saliency maps for  {id} (class {class_name})...")
        model,encoder = models.create_model(model_name,device,len(config.class_names))
        model.load_state_dict(torch.load('checkpoint/'+model_name+'/best_model_state.ckpt',map_location=config.device))
        model.name=model_name

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(pytorch_total_params)

        model.half() #fp16
        opt_model = model#torch.compile(model) #, mode="max-autotune", fullgraph=True)
        
        generate_visualizations(opt_model,encoder,dataloader, device,config.inverse_normalize_transform,id,class_name,k_clustering)
      
        

       

        