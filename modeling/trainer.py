import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from loss import FocalLoss
from scripts.utils import build_label_to_path, lca_depth


class FocalTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.class_weights = class_weights.cuda() if torch.cuda.is_available() else class_weights
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss_foc = FocalLoss(alpha=self.class_weights, gamma=2)
        loss = loss_foc(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss
    
    
class WeightedCETrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.class_weights = class_weights.cuda() if torch.cuda.is_available() else class_weights
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss_ce = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_ce(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss
    
    
class HierarchicalFocalTrainer(Trainer):
    def __init__(
        self,
        category_tree_df,
        mapping, # list/array: mapping[class_idx] -> real cat_id
        gamma=2.0,
        lambda_hier=1.0,
        class_weights=None,
        diff_dist=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.label_to_path = build_label_to_path(category_tree_df)
        self.label_to_depth = {
            cat_id: len(path) for cat_id, path in self.label_to_path.items()
        }

        self.mapping = mapping
        
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            if torch.cuda.is_available():
                class_weights = class_weights.cuda()
        self.focal_loss_fn = FocalLoss(alpha=class_weights, gamma=gamma, reduction='none')

        self.lambda_hier = lambda_hier

    def compute_distance_multiplier(self, logits, labels):
        """
        dist_i = sum_{c} [ p_i(c) * dist(mapping[labels[i]], mapping[c]) ]
        multiplier_i = 1 + lambda_hier * dist_i
        """
        device = logits.device
        batch_size, num_classes = logits.shape
        
        probs = F.softmax(logits, dim=1)  # [batch_size, num_classes]
        distance_per_item = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            true_idx = labels[i].item()
            true_cat_id = self.mapping[true_idx]
            path_y = self.label_to_path[true_cat_id]
            depth_y = self.label_to_depth[true_cat_id]
            
            dist_sum = 0.0
            for c in range(num_classes):
                cat_id_c = self.mapping[c]
                path_c = self.label_to_path[cat_id_c]
                depth_lca = lca_depth(path_y, path_c)
                dist_val = max(0, depth_y - depth_lca)
                dist_sum += probs[i, c].item() * dist_val
            
            distance_per_item[i] = dist_sum
        
        multiplier = 1.0 + self.lambda_hier * distance_per_item
        return multiplier

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')  # [batch_size, num_classes]

        focal_losses = self.focal_loss_fn(logits, labels)  # [batch_size]

        if diff_dist:
            distance_multiplier = self.compute_distance_multiplier(logits, labels)
        else:
            with torch.no_grad():
                distance_multiplier = self.compute_distance_multiplier(logits, labels)   

        total_loss_per_item = focal_losses * distance_multiplier

        loss = total_loss_per_item.mean()
        
        return (loss, outputs) if return_outputs else loss