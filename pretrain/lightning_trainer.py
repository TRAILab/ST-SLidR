from cmath import cos
import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss
from pytorch_lightning.utilities import rank_zero_only
import torch.nn.functional as F
from torchvision import transforms
from torch import nn


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images
        self._config = config
        self.losses_enabled = config["criteria"]["losses"]["enabled"]
        assert len(self.losses_enabled) > 0, "At least one loss is needed"
        assert len(self.losses_enabled) == len(set(self.losses_enabled)), "Enabled losses list must be unique"
        self.losses_params = config["criteria"]["losses"]["params"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["criteria"]["num_matches"]
        self.batch_size = config["trainer"]["batch_size"]
        self.num_epochs = config["trainer"]["num_epochs"]
        self.superpixel_size = config["superpixels"]["size"]
        self.epoch = 0
        if config["trainer"]["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["trainer"]["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["criteria"]["NCE_temperature"])
        self.working_dir = config["working_dir"]
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()),
            lr=self._config["solver"]["lr"],
            momentum=self._config["solver"]["momentum"],
            dampening=self._config["solver"]["dampening"],
            weight_decay=self._config["solver"]["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model_points(sparse_input).F
        self.model_images.eval()
        self.model_images.decoder.train()
        output_images = self.model_images(batch["input_I"])
        assert 'embeddings' in output_images

        del batch["sinput_F"]
        del sparse_input
        # each loss is applied independtly on each GPU
        losses = [
            getattr(self, loss_name)(batch, output_points, output_images, loss_params=self.losses_params.get(loss_name))
            for loss_name in self.losses_enabled
        ]
        loss = torch.mean(torch.stack(losses))

        torch.cuda.empty_cache()
        self.log(
            "pretrain/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def loss(self, batch, output_points, output_images, loss_params=None):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)

    
    # get superpoint, superpixel embeddings
    def get_k_q(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]
        output_images = output_images['embeddings']
        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        return k, q, one_hot_P, one_hot_I

    
    def loss_superpixels_average(self, batch, output_points, output_images, loss_params=None):
        k, q, _, _= self.get_k_q(batch, output_points, output_images)
        occupied_superpoints_mask = torch.where(k[:, 0] != 0)
        k = k[occupied_superpoints_mask]
        q = q[occupied_superpoints_mask]

        return self.criterion(k, q)

    
    # rename loss_superpixels_feat_similarity_average to loss_superpixel_stslidr
    def loss_superpixel_stslidr(self, batch, output_points, output_images, loss_params):
        k, q, one_hot_P, one_hot_I = self.get_k_q(batch, output_points, output_images)
        occupied_superpoints_mask = torch.where(k[:, 0] != 0)
        k = k[occupied_superpoints_mask]
        q = q[occupied_superpoints_mask]

        assert 'features' in output_images
        image_features = output_images['features']
        ### Get superpixel to superpixel similarity based on image features
        sp_features = self.get_superpixel_features(batch, image_features)
        cosine_sim = self.get_superpixel_cosine_similarity(batch, sp_features, occupied_superpoints_mask, one_hot_P, 
                                                      is_batch_level=True)        

        min_quantile = loss_params['min_quantile'] if 'min_quantile' in loss_params else 0.0
        max_quantile = loss_params['max_quantile'] if 'max_quantile' in loss_params else 1.0
        return self.criterion(k, q, cosine_sim, min_quantile, max_quantile)

    # use feature similarity as a proxy for class count to reweight semantic slidr loss 
    # rename loss_superpixels_bg_feat_reweighting_semSlidr to 
    def loss_superpixel_stslidr_classbalance(self, batch, output_points, output_images, loss_params):
        k, q, one_hot_P, one_hot_I = self.get_k_q(batch, output_points, output_images)
        # filter empty superpoints
        occupied_superpoints_mask = torch.where(k[:, 0] != 0)
        k = k[occupied_superpoints_mask]
        q = q[occupied_superpoints_mask]

        # get feature similarity
        assert 'features' in output_images
        image_features = output_images['features']
        ### Get superpixel to superpixel similarity based on image features
        sp_features = self.get_superpixel_features(batch, image_features)
        # cosine similarity for negative sample weighting - semantic slidr
        cosine_sim = self.get_superpixel_cosine_similarity(batch, sp_features, occupied_superpoints_mask, one_hot_P, 
                                                           is_batch_level=True)
        # cosine similarity for class balancing
        cosine_sim_weight_balance = self.get_superpixel_cosine_similarity(batch, sp_features, occupied_superpoints_mask, one_hot_P, 
                                                           is_batch_level=True)
        # remove superpoints and superpixels that have zero pretrained feature vectors
        nonzero_superpixel_feat_mask = torch.where(torch.sum(cosine_sim, dim=1)!= 0.0)
        # filter similarity where rows and columns are zeros
        cosine_sim = torch.index_select(cosine_sim, dim=0, index=nonzero_superpixel_feat_mask[0])
        cosine_sim = torch.index_select(cosine_sim, dim=1, index=nonzero_superpixel_feat_mask[0])
        cosine_sim_weight_balance = torch.index_select(cosine_sim_weight_balance, dim=0, index=nonzero_superpixel_feat_mask[0])
        cosine_sim_weight_balance = torch.index_select(cosine_sim_weight_balance, dim=1, index=nonzero_superpixel_feat_mask[0])
        # filter zero superpixel features
        k = k[nonzero_superpixel_feat_mask]
        q = q[nonzero_superpixel_feat_mask]

        # define un-reduced cross entropy loss
        unreduced_cross_entropy = nn.CrossEntropyLoss(reduction='none')
        # compute semantic slidr cross entropy loss
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.criterion.temperature)
        # read quantiles
        min_quantile = loss_params['min_quantile'] if 'min_quantile' in loss_params else 0.0
        max_quantile = loss_params['max_quantile'] if 'max_quantile' in loss_params else 1.0
        # weight using sematic similarity 
        weight = 1.0 - cosine_sim
        # diagonal elements are positive
        if min_quantile > 0.0 or max_quantile < 1.0:
            min_quantile_value = torch.quantile(weight ,q=min_quantile, dim=1)
            max_quantile_value = torch.quantile(weight ,q=max_quantile, dim=1)
            weight[weight<min_quantile_value.view(-1, 1)] = 0.0
            weight[weight>max_quantile_value.view(-1, 1)] = 0.0
            non_zero_indices = torch.nonzero(weight, as_tuple=True)
            weight[non_zero_indices] = 1.0
        weight.fill_diagonal_(1.0)
        
        out = out * weight
        out = out.contiguous()
        loss = unreduced_cross_entropy(out, target)

        # return a weight for each anchor based on feature similarity
        weight_per_anchor = self.get_anchor_weights(cosine_sim_weight_balance, loss_params)
        # background reweighting and normalization
        loss_reweighted = torch.dot(loss, weight_per_anchor) / weight_per_anchor.sum()
        return loss_reweighted

    # use feature similarity as a proxy for class count to reweight slidr loss
    def loss_superpixels_bg_feat_reweighting(self, batch, output_points, output_images, loss_params):
        k, q, one_hot_P, one_hot_I = self.get_k_q(batch, output_points, output_images)
        occupied_superpoints_mask = torch.where(k[:, 0] != 0)
        k = k[occupied_superpoints_mask]
        q = q[occupied_superpoints_mask]
        # ensure image features exist
        assert 'features' in output_images
        image_features = output_images['features']
        ### Get superpixel to superpixel similarity based on image features
        sp_features = self.get_superpixel_features(batch, image_features)
        cosine_sim = self.get_superpixel_cosine_similarity(batch, sp_features, occupied_superpoints_mask, one_hot_P, 
                                                      is_batch_level=True)
      
        # remove superpoints and superpixels that have zero pretrained feature vectors
        nonzero_superpixel_feat_mask = torch.where(torch.sum(cosine_sim, dim=1)!= 0.0)
        # filter similarity where rows and columns are zeros
        cosine_sim = torch.index_select(cosine_sim, dim=0, index=nonzero_superpixel_feat_mask[0])
        cosine_sim = torch.index_select(cosine_sim, dim=1, index=nonzero_superpixel_feat_mask[0])
        # filter superpoints/superpixels
        k = k[nonzero_superpixel_feat_mask]
        q = q[nonzero_superpixel_feat_mask]
 
        # define un-reduced cross entropy loss
        unreduced_cross_entropy = nn.CrossEntropyLoss(reduction='none')
        # compute cross entropy loss
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.criterion.temperature)
        out = out.contiguous()
        loss = unreduced_cross_entropy(out, target)
        
        # return a weight for each anchor based on feature similarity
        weight_per_anchor = self.get_anchor_weights(cosine_sim, loss_params)
        # background reweighting and normalization
        loss_reweighted = torch.dot(loss, weight_per_anchor) / weight_per_anchor.sum()
        return loss_reweighted

    # return a weight for each anchor based on feature similarity
    def get_anchor_weights(self, cosine_sim, loss_params):
        votes_per_anchor = self.get_votes(cosine_sim, loss_params)
        class_balancing_method = 'per_anchor_similarity'
        if class_balancing_method == 'per_anchor_similarity':
            anchor_weights = 1.0 - votes_per_anchor
        elif class_balancing_method == 'fg_bg_buckets':
            percentile = loss_params['background_percentile']
            assert percentile >= 0.0 and percentile <= 1.0
            num_anchors = cosine_sim.shape[0]
            k = int(num_anchors * percentile)
            # get  indices of top k anchors (background anchors)
            values, background_indices = votes_per_anchor.topk(k)
            indices = torch.zeros(num_anchors, dtype=bool, device=cosine_sim.device)
            indices[background_indices] = True
            # balance weights
            anchor_weights = torch.ones(num_anchors, dtype=torch.float32, device=cosine_sim.device)
            # bg weight
            anchor_weights[indices] = 1.0 - percentile
            # fg weight
            anchor_weights[~indices] = percentile
        else:
            raise NotImplementedError("Class balancing weighting function not implemented !")

        return anchor_weights

    # pre-process cosine similarity and compute votes 
    def get_votes(self, cosine_sim, loss_params):
        minimum_threshold = loss_params['loss_balance_noise_threshold']
        similarity = cosine_sim.clone()
        similarity[similarity<minimum_threshold] = 0.0
        votes_per_anchor = similarity.sum(dim=1)
        votes_per_anchor = votes_per_anchor - votes_per_anchor.min()
        votes_per_anchor_normalized = votes_per_anchor / votes_per_anchor.max()
        return votes_per_anchor_normalized


    def get_scene_level_featsim(self, superpixel_features_normalized, superpoint_coordinates):
        assert superpixel_features_normalized.shape[0] == superpoint_coordinates.shape[0]
        batch_size = int(superpoint_coordinates[:,0].max()) + 1
        # Compute batch similarity matrix from superpixel-to-superpixel for each scene
        batch_similarity = torch.empty(0,0, device=superpixel_features_normalized.device)
        for bt_idx in range(0, batch_size):
            curr_superpixel_features = superpixel_features_normalized[torch.where(superpoint_coordinates[:,0]==bt_idx)]
            scene_similarity = curr_superpixel_features @ curr_superpixel_features.T
            batch_similarity = torch.block_diag(batch_similarity, scene_similarity)
        return batch_similarity

    def get_superpixel_cosine_similarity(self, batch, sp_features, occupied_superpoints_mask, one_hot_P, is_batch_level):
        # # mask out superpixels corresponding to empty superpoints
        sp_features = sp_features[occupied_superpoints_mask]
        # sp to sp compute similarity
        sp_features_normalized = F.normalize(sp_features, p=2, dim=1)
        # compute batch-level or scene-level feature similarity 
        if is_batch_level:
            cosine_sim = sp_features_normalized @ sp_features_normalized.T
        else:
            # compute scene-level feature similarity
            # associate superpoints with batch id by averaging points
            pairing_points = batch["pairing_points"] 
            superpoint_coordinates = one_hot_P @ batch["sinput_C"][pairing_points].float()
            # (batch_id, superpoint_meanX, superpoint_meanY, superpoint_meanZ)
            superpoint_coordinates = superpoint_coordinates / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
            # filter empty superpoints, similar to k, q and sp_features
            superpoint_coordinates = superpoint_coordinates[occupied_superpoints_mask]
            # convert batch idx from float to int resulting from coordinates pooling
            batch_idx_tensor = torch.round(superpoint_coordinates[:,0]).type(torch.uint8)
            superpoint_coordinates[:,0] = batch_idx_tensor 
            cosine_sim = self.get_scene_level_featsim(sp_features_normalized, superpoint_coordinates)
        return cosine_sim
    
    # get superpixels features based on image pretrained features
    def get_superpixel_features(self, batch, image_features):
        superpixels = batch["superpixels"]
        # Down sample to match feature map dimensions
        size = (image_features.shape[2], image_features.shape[3])
        interpolation = transforms.InterpolationMode.NEAREST # nearest neigbour
        downsample_transform = transforms.Compose([transforms.Resize(size, interpolation=interpolation)])
        superpixels = downsample_transform(superpixels)
        # creates a unique ID for each superpixel in a batch (adds self.superpixel_size to each set of superpixels from each image)
        superpixels = (
            torch.arange(
                0,
                image_features.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )

        superpixels_I = superpixels.flatten()
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        # Sum all pixel features with the same super pixel id, then divide by sum.
        sp_features = one_hot_I @ image_features.permute(0, 2, 3, 1).flatten(0, 2)
        sp_features = sp_features / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)
        return sp_features  

    def loss_ppkt(self, batch, output_points, output_images, loss_params=None):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        output_images = output_images['embeddings']
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)

    def loss_ppkt_feat_similarity_average(self, batch, output_points, output_images, loss_params):
        assert 'features' in output_images

        image_features = output_images['features']
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        output_embeddings = output_images['embeddings']

        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_embeddings.permute(0, 2, 3, 1)[m]

        # get interpolated pretrained features at sampled pixels
        sampled_image_features = self.get_pixel_features(image_features, m).view(k.shape[0],-1)

        ### Get pixel to pixel similarity based on image features
        num_image_per_scene = int(image_features.shape[0] / self.batch_size)
        cosine_sim = self.get_pixel_cosine_similarity(sampled_image_features, m[0], num_image_per_scene, is_batch_level=True)

        # filter/scale/weight similarity
        if 'featsim_min_threshold' == loss_params['weighting_function']:
            assert loss_params['featsim_min_threshold'] >= 0.0 and loss_params['featsim_min_threshold'] <= 1.0
            cosine_sim[(cosine_sim <= loss_params['featsim_min_threshold']) &
                        ~torch.eye(len(cosine_sim), dtype=torch.bool, device=cosine_sim.device)] = 0.0
        elif 'featsim_exponent' == loss_params['weighting_function']:
            assert loss_params['featsim_exponent'] >= 1.0
            cosine_sim_non_diag_mask = ~torch.eye(len(cosine_sim), dtype=torch.bool, device=cosine_sim.device)
            cosine_sim[cosine_sim_non_diag_mask] = torch.pow(cosine_sim[cosine_sim_non_diag_mask], loss_params['featsim_exponent'])
        else:
            raise NotImplementedError("Superpixel similarity weighting function not implemented !")
        
        min_quantile = loss_params['min_quantile'] if 'min_quantile' in loss_params else 0.0
        max_quantile = loss_params['max_quantile'] if 'max_quantile' in loss_params else 1.0
        return self.criterion(k, q, cosine_sim, min_quantile, max_quantile)

    def loss_ppkt_feat_reweighting_semPPKT(self, batch, output_points, output_images, loss_params):
        assert 'features' in output_images

        image_features = output_images['features']
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        output_embeddings = output_images['embeddings']

        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_embeddings.permute(0, 2, 3, 1)[m]

        # get interpolated pretrained features at sampled pixels
        sampled_image_features = self.get_pixel_features(image_features, m).view(k.shape[0],-1)

        ### Get pixel to pixel similarity based on image features
        num_image_per_scene = int(image_features.shape[0] / self.batch_size)
        # cosine similarity for negative sample weighting - semantic PPKT
        cosine_sim = self.get_pixel_cosine_similarity(sampled_image_features, m[0], num_image_per_scene, 
                                                      is_batch_level=True)
        # # cosine similarity for class balancing
        cosine_sim_weight_balance = self.get_pixel_cosine_similarity(sampled_image_features, m[0], num_image_per_scene, 
                                                      is_batch_level=True)
        
        # define un-reduced cross entropy loss
        unreduced_cross_entropy = nn.CrossEntropyLoss(reduction='none')
        # compute semantic slidr cross entropy loss
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.criterion.temperature)
        # read quantiles
        min_quantile = loss_params['min_quantile'] if 'min_quantile' in loss_params else 0.0
        max_quantile = loss_params['max_quantile'] if 'max_quantile' in loss_params else 1.0
        # weight using sematic similarity 
        weight = 1.0 - cosine_sim
        # diagonal elements are positive
        if min_quantile > 0.0 or max_quantile < 1.0:
            min_quantile_value = torch.quantile(weight ,q=min_quantile, dim=1)
            max_quantile_value = torch.quantile(weight ,q=max_quantile, dim=1)
            weight[weight<min_quantile_value.view(-1, 1)] = 0.0
            weight[weight>max_quantile_value.view(-1, 1)] = 0.0
            non_zero_indices = torch.nonzero(weight, as_tuple=True)
            weight[non_zero_indices] = 1.0
        weight.fill_diagonal_(1.0)
        
        out = out * weight
        out = out.contiguous()
        loss = unreduced_cross_entropy(out, target)

        # return a weight for each anchor based on feature similarity
        weight_per_anchor = self.get_anchor_weights(cosine_sim_weight_balance, loss_params)
        # background reweighting and normalization
        loss_reweighted = torch.dot(loss, weight_per_anchor) / weight_per_anchor.sum()
        return loss_reweighted
    
    def get_pixel_cosine_similarity(self, pixel_features, sampled_indices, num_image_per_scene, is_batch_level=False):
        pixel_features_normalized = F.normalize(pixel_features, p=2, dim=1)
        # compute batch-level or scene-level feature similarity 
        if is_batch_level:
            cosine_sim = pixel_features_normalized @ pixel_features_normalized.T
        else:
            cosine_sim = torch.empty(0,0, device=pixel_features.device)
            for bt_idx in range(0, self.batch_size):
                curr_pixel_features = pixel_features_normalized[torch.where((sampled_indices/num_image_per_scene).long()==bt_idx)]
                scene_similarity = curr_pixel_features @ curr_pixel_features.T
                cosine_sim = torch.block_diag(cosine_sim, scene_similarity)
        return cosine_sim

    def get_pixel_features(self, image_features, sampled_pixels):
        # get feature maps
        sampled_image_features = image_features[sampled_pixels[0]]

        # construct keypoint image for nn.grid_sample
        keypoint_img = torch.cat([sampled_pixels[1].view(-1,1),sampled_pixels[2].view(-1,1)],axis=1).unsqueeze(1).float()
        keypoint_pixels = self.convert_to_normalized_range((self._config["model"]["image"]["crop_size"][0],
                                                            self._config["model"]["image"]["crop_size"][1]), keypoint_img)
        
        # sample interpolated features
        sampled_features = nn.functional.grid_sample(input=sampled_image_features, grid=keypoint_pixels.unsqueeze(1), mode='bilinear', 
                                                        padding_mode='zeros', align_corners=True)
        return sampled_features

    def convert_to_normalized_range(self, image_range, sampled_pixel):
        keypoint_pixel = sampled_pixel.clone()

        B = keypoint_pixel.shape[0]
        old_h_img_range, old_w_img_range = image_range
        old_min = 0
        
        new_max = 1.0
        new_min = -1.0
        new_h_norm_range = new_max - new_min
        new_w_norm_range = new_max - new_min
        for b in range(B):
            u = keypoint_pixel[b, :, 0]
            v = keypoint_pixel[b, :, 1]
            keypoint_pixel[b, :, 0] = (((u - old_min) * new_w_norm_range) / old_w_img_range) + new_min
            keypoint_pixel[b, :, 1] = (((v - old_min) * new_h_norm_range) / old_h_img_range) + new_min
        
        return keypoint_pixel
    

    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            filepath = os.path.join(self.working_dir, "model.pt")
            self.save(filepath)
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model_points(sparse_input).F
        self.model_images.eval()
        output_images = self.model_images(batch["input_I"])
        assert 'embeddings' in output_images

        losses = [
            getattr(self, loss_name)(batch, output_points, output_images, loss_params=self.losses_params.get(loss_name))
            for loss_name in self.losses_enabled
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "pretrain/val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        
        return loss

    @rank_zero_only
    def save(self, filepath):
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_images": self.model_images.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            filepath,
        )