import torch
import os
import argparse
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()
    return args

def remove_associate_keypoint_heads(state_dict, weight_names, num_heads=5):
    """Remove associate keypoint heads from state dict."""
    keys_to_remove = []
    for j in range(num_heads):
        for tensor_name in weight_names:
            associate_key = tensor_name.replace('keypoint_head', f'associate_keypoint_heads.{j}')
            if associate_key in state_dict:
                keys_to_remove.append(associate_key)

    for key in keys_to_remove:
        state_dict.pop(key)

def remove_expert_layers(state_dict):
    """Remove all expert layers from state dict."""
    keys_to_remove = [key for key in state_dict.keys() if 'expert' in key]
    for key in keys_to_remove:
        state_dict.pop(key)

def extract_experts(state_dict):
    """Extract expert layers from state dict."""
    experts = {}
    for key, value in state_dict.items():
        if 'mlp.experts' in key:
            experts[key] = value
    return experts

def concatenate_experts(state_dict, experts, target_expert, keys):
    """Concatenate expert weights to fc2 layers."""
    for key in keys:
        if 'mlp.fc2' in key:
            expert_key = key.replace('fc2.', f'experts.{target_expert}.')
            if expert_key in experts:
                value = state_dict[key]
                value = torch.cat([value, experts[expert_key]], dim=0)
                state_dict[key] = value

def create_dataset_checkpoint(base_ckpt, experts, target_expert, weight_names, num_keypoints=None, associate_head_idx=None):
    """Create a checkpoint for a specific dataset."""
    new_ckpt = copy.deepcopy(base_ckpt)
    state_dict = new_ckpt['state_dict']
    keys = base_ckpt['state_dict'].keys()

    # Concatenate experts
    concatenate_experts(state_dict, experts, target_expert, keys)

    # For non-COCO datasets, copy associate keypoint head to main keypoint head
    if associate_head_idx is not None:
        for tensor_name in weight_names:
            associate_key = tensor_name.replace('keypoint_head', f'associate_keypoint_heads.{associate_head_idx}')
            if associate_key in state_dict:
                state_dict[tensor_name] = state_dict[associate_key]

        # Adjust final layer dimensions for specific number of keypoints
        if num_keypoints is not None:
            for tensor_name in ['keypoint_head.final_layer.weight', 'keypoint_head.final_layer.bias']:
                if tensor_name in state_dict:
                    state_dict[tensor_name] = state_dict[tensor_name][:num_keypoints]

    # Clean up unnecessary parameters
    remove_associate_keypoint_heads(state_dict, weight_names)
    remove_expert_layers(state_dict)

    return new_ckpt

def main():
    args = parse_args()

    if args.target is None:
        args.target = '/'.join(args.source.split('/')[:-1])

    # Load checkpoint and extract experts
    ckpt = torch.load(args.source, map_location='cpu')
    experts = extract_experts(copy.deepcopy(ckpt['state_dict']))

    # Define weight names for keypoint heads
    weight_names = [
        'keypoint_head.deconv_layers.0.weight',
        'keypoint_head.deconv_layers.1.weight',
        'keypoint_head.deconv_layers.1.bias',
        'keypoint_head.deconv_layers.1.running_mean',
        'keypoint_head.deconv_layers.1.running_var',
        'keypoint_head.deconv_layers.1.num_batches_tracked',
        'keypoint_head.deconv_layers.3.weight',
        'keypoint_head.deconv_layers.4.weight',
        'keypoint_head.deconv_layers.4.bias',
        'keypoint_head.deconv_layers.4.running_mean',
        'keypoint_head.deconv_layers.4.running_var',
        'keypoint_head.deconv_layers.4.num_batches_tracked',
        'keypoint_head.final_layer.weight',
        'keypoint_head.final_layer.bias'
    ]

    # Create COCO checkpoint (expert 0)
    coco_ckpt = create_dataset_checkpoint(ckpt, experts, target_expert=0, weight_names=weight_names)
    torch.save(coco_ckpt, os.path.join(args.target, 'coco.pth'))

    # Create checkpoints for other datasets
    # (dataset_name, num_keypoints, associate_head_idx)
    dataset_configs = [
        ('aic', 14, 0),
        ('mpii', 16, 1),
        ('ap10k', 17, 2),
        ('apt36k', 17, 3),
        ('wholebody', 133, 4)
    ]

    for name, num_kpts, associate_idx in dataset_configs:
        target_expert = associate_idx + 1

        # Check if expert exists
        expert_exists = any(f'experts.{target_expert}.' in key for key in experts.keys())
        if not expert_exists:
            print(f"Warning: Expert {target_expert} not found for dataset {name}. Skipping.")
            break

        dataset_ckpt = create_dataset_checkpoint(
            ckpt, experts, target_expert, weight_names,
            num_keypoints=num_kpts, associate_head_idx=associate_idx
        )
        torch.save(dataset_ckpt, os.path.join(args.target, f'{name}.pth'))

if __name__ == '__main__':
    main()
