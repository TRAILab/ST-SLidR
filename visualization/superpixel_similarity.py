import os
import argparse
import torch
from utils.read_config import generate_config
from pretrain.model_builder import make_model
from pretrain.dataloader_nuscenes import NuScenesMatchDataset, minkunet_collate_pair_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import torch.nn.functional as F

# superpixels: original superpixel image dimensions (dtype = uint8)
# index: ID of superpixel to visualize
# computed_similarity: any measure of similarity ranging from 0.0 1.0
def similarity_map(superpixels: torch.Tensor, index: int, computed_similarity: torch.Tensor):
    superpixels = torch.squeeze(superpixels)
    mask = torch.zeros(superpixels.shape, dtype=torch.uint8)
    similarity = computed_similarity[index,:]
    number_of_superpixels = computed_similarity.shape[0]
    for superpixel_idx in range(0, number_of_superpixels):
       mask[torch.where(superpixels==superpixel_idx)] = int(255.0 * similarity[superpixel_idx].item())
    return mask

# visualize SetSim saliency of pretrained features
#https://arxiv.org/pdf/2107.08712.pdf
def saliency_map(extracted_features: torch.tensor):
    # Attention saliency map - sum along C
    extracted_features = torch.abs(extracted_features)
    attention = torch.sum(torch.squeeze(extracted_features), dim=0)
    delta = attention.max() - attention.min()
    attention = (attention - attention.min()) / delta
    return attention

def main():
    """
    Code for launching the pretraining
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/image_features.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    args = parser.parse_args()
    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path


    # load image model, add upsampling layer and move to GPU
    model_points, model_images = make_model(config)
    del model_points
    #upsample = torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
    model = model_images.encoder
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model.eval()
    # load dataset
    dataset = NuScenesMatchDataset(phase = 'mini_train', config=config)
    dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config["num_threads"],
            collate_fn=minkunet_collate_pair_fn)
    
    images = []
    for idx, sample in enumerate(dataloader):
        if idx < 500:
            images.append(sample)
        if idx >= 500:
            break
    
    superpixel_size = config['superpixel_size']
    p2t_transform = transforms.PILToTensor() 
    t2p_transform = transforms.ToPILImage()
    STRIDE_RATIO = 4
    images = images[136:]
    with torch.no_grad():
        for idx, sample in enumerate(images):
            # load image and superpixels
            image = sample['input_I']
            im = transforms.ToPILImage()(torch.squeeze(image)).convert("RGB")
            superpixels_orig = torch.squeeze(sample['superpixels'])
            print('\nSuperpixel tensor shape {}'.format(superpixels_orig.shape))
            super_im = t2p_transform(superpixels_orig)
            size = (superpixels_orig.shape[0]//STRIDE_RATIO, superpixels_orig.shape[1]//STRIDE_RATIO)
            interpolation = 0 # nearest neigbour
            # downsample superpixel image to match image feature map
            transform_pillow_resize = transforms.Compose([transforms.Resize(size, interpolation=interpolation)])
            super_pillow_resize = transform_pillow_resize(super_im)
            pil_to_tensor = p2t_transform(super_pillow_resize) 
            print('\nSuperpixel tensor after downsampling {}'.format(pil_to_tensor.shape))
            superpixels = pil_to_tensor
            superpixels_I = superpixels.flatten()
            total_pixels = superpixels_I.shape[0]
            idx_I = torch.arange(total_pixels, device=superpixels.device)
            # sparse tensor of 1's and 0's 
            # Num of superpixels * Num of images pixels
            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * superpixel_size, total_pixels)
            )
            print('\nShape of one hot superPixelID X imagePixel {}'.format(one_hot_I.shape))
            # Extract image features
            output_images = model(image)
            print('\nShape of output images {}'.format(output_images.shape))
            # Compute mean feature vector for each superpixel - Num of superpixels * image features (2048)
            spixel_feature = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
            spixel_feature = spixel_feature / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)
            print('\nShape of superPixelID X mean image features {}'.format(spixel_feature.shape))

            
            # remove unused superpixels
            NUM_VALID_SUPERPIXELS = superpixels_orig.max()
            spixel_feature = spixel_feature[0:NUM_VALID_SUPERPIXELS]

            # Unnormalized similarity - Num of superpixels * Num of superpixels
            raw_similarity = spixel_feature @ spixel_feature.T
            out, inds = torch.max(raw_similarity, dim=1)
            raw_similarity_row_norm = raw_similarity/(out + 1e-6)
            
            # Normalized 
            l2norm_spixel_feature = F.normalize(spixel_feature, p=2, dim=1)
            norm_similarity = l2norm_spixel_feature @ l2norm_spixel_feature.T

            # Centered around feature mean
            spixel_feature_mean = torch.mean(spixel_feature, dim=0)
            centered_spixel_feature = spixel_feature - spixel_feature_mean
            centered_similarity = centered_spixel_feature @ centered_spixel_feature.T

            # Euclidean similarity
            distance_similarity = torch.cdist(spixel_feature, spixel_feature)
            out, inds = torch.max(distance_similarity, dim=1)
            distance_similarity_row_norm = 1.0 - distance_similarity/(out + 1e-6)

            # visualize superpixel similarity
            mask = similarity_map(superpixels=superpixels_orig, index=0, computed_similarity=norm_similarity)

            # SetSim saliency
            attention_mask = saliency_map(extracted_features=output_images)


if __name__ == "__main__":
    main()
	