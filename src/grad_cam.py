
import torch
import torch.nn.functional as F


def grad_cam(model, image, target_class, target_layer, normalize=True, reshape_transform=None):
    """
    Calculate Grad-CAM for one image.

    image: Tensor of shape [1, 3, 32, 32]
    target_class: CIFAR-10 class index (0 to 9)
    target_layer: convolutional layer to explain
    normalize:
        True  -> heatmap normalized to [0, 1]
        False -> raw nonnegative Grad-CAM values

    Returns:
        Heatmap of shape [1, 1, 32, 32]
    """

    model.eval()
    activations = []

    # Capture feature maps from the selected layer
    def forward_hook(module, inputs, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        # Forward pass: obtain the 10 class scores
        logits = model(image)

        # Select the class score we want to explain
        score = logits[:, target_class].sum()

        # Gradients of that score with respect to feature maps
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=activations[0],
        )[0]

        feature_maps = activations[0]
        if reshape_transform is not None:
            feature_maps = reshape_transform(feature_maps)
            gradients = reshape_transform(gradients)

        # Average gradients spatially to get channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of the feature maps
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)

        # Keep positive contributions
        cam = F.relu(cam)

        # Resize from feature-map size to input-image size
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Normalize only when requested
        if normalize:
            cam_min = cam.amin(dim=(2, 3), keepdim=True)
            cam_max = cam.amax(dim=(2, 3), keepdim=True)

            cam = (cam - cam_min) / (
                cam_max - cam_min + 1e-8
            )

        return cam.detach()

    finally:
        handle.remove()
