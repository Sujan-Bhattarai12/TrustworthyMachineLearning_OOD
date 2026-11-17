"""CIFAR-10 classifier adapters for pinned published weights and local checkpoints.

Published models preserve their trained heads; dropout is never injected.
Local dropout-head training helpers remain available as an optional alternative.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

ARCHITECTURES = ('resnet50', 'vit_b_16', 'swin_t')
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class PixelClassifier(nn.Module):
    def __init__(self, backbone, image_size, mean, std):
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, image):
        image = F.interpolate(image, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False, antialias=True)
        return self.backbone((image - self.mean) / self.std)


def build_classifier(architecture, image_size=224, mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225), head_dropout=0.2):
    if architecture not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture: {architecture}')
    if image_size != 224:
        raise ValueError('This configuration supports 224-pixel classifiers.')
    if not 0 < head_dropout < 1:
        raise ValueError('A trained dropout head with 0 < p < 1 is required.')
    net = getattr(models, architecture)(weights=None)
    attr = {'resnet50': 'fc', 'vit_b_16': 'heads', 'swin_t': 'head'}[architecture]
    layer = getattr(net, attr)
    width = layer.head.in_features if architecture == 'vit_b_16' else layer.in_features
    setattr(net, attr, nn.Sequential(nn.Dropout(head_dropout), nn.Linear(width, 10)))
    return PixelClassifier(net, image_size, mean, std)


def load_classifier(path, architecture, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required = {'architecture', 'state_dict', 'class_names', 'image_size', 'mean',
                'std', 'head_dropout', 'dropout_trained'}
    if not required.issubset(checkpoint):
        raise ValueError(f'{path}: missing checkpoint metadata {sorted(required - set(checkpoint))}')
    if checkpoint['architecture'] != architecture or checkpoint['class_names'] != CIFAR10_CLASSES:
        raise ValueError('Checkpoint architecture or CIFAR-10 label order mismatch.')
    if checkpoint['dropout_trained'] is not True:
        raise ValueError('Checkpoint must have been trained/fine-tuned with dropout enabled.')
    model = build_classifier(architecture, **{k: checkpoint[k] for k in
                             ('image_size', 'mean', 'std', 'head_dropout')})
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval()


def attribution_layer(model, architecture):
    net = model.backbone
    if getattr(model, 'backend', None) == 'transformers':
        if architecture == 'vit_b_16':
            def reshape_tokens(x):
                return x[:, 1:].transpose(1, 2).reshape(x.shape[0], x.shape[2], 14, 14)
            return net.vit.encoder.layer[-1].layernorm_before, reshape_tokens
        def reshape_swin(x):
            return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], 7, 7)
        return net.swin.encoder.layers[-1].blocks[-1].layernorm_before, reshape_swin
    if architecture == 'resnet50':
        return net.layer4[-1].conv3, None
    if architecture == 'vit_b_16':
        # Before the final attention operation, patch tokens influence the class token.
        def reshape(x):
            return x[:, 1:].transpose(1, 2).reshape(x.shape[0], x.shape[2], 14, 14)
        return net.encoder.layers[-1].ln_1, reshape
    return net.features[-1][-1].norm1, lambda x: x.permute(0, 3, 1, 2)


class PublishedPixelClassifier(PixelClassifier):
    """Differentiable preprocessing of float [0,1] pixels; no quantization."""
    def __init__(self, backbone, image_size, mean, std, backend, interpolation='bilinear'):
        super().__init__(backbone, image_size, mean, std)
        self.backend = backend
        self.interpolation = interpolation

    def forward(self, image):
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size),
                                  mode=self.interpolation, align_corners=False, antialias=True)
        # Bicubic overshoots are clipped as in valid-image preprocessing.
        image = (image.clamp(0, 1) - self.mean) / self.std
        output = self.backbone(image)
        return output.logits if self.backend == 'transformers' else output


def load_published_classifier(directory, architecture, device='cpu'):
    """Load pinned local HF downloads without executing repository Python code."""
    import json
    import hashlib
    from pathlib import Path
    from safetensors.torch import load_file
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    provenance = json.loads((directory / 'provenance.json').read_text())
    for name, expected_digest in provenance['sha256'].items():
        digest = hashlib.sha256()
        with (directory / name).open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError(f'Checkpoint file checksum mismatch: {name}')
    weight_path = directory / provenance['weights']
    state = (load_file(str(weight_path)) if weight_path.suffix == '.safetensors'
             else torch.load(weight_path, map_location='cpu', weights_only=True))
    backend = provenance['backend']
    if backend == 'cifar_resnet':
        if architecture != 'resnet50' or config.get('num_classes') != 10:
            raise ValueError('Not a CIFAR-10 ResNet-50 checkpoint.')
        net = models.resnet50(weights=None, num_classes=10)
        # The published CIFAR architecture uses a small stem without max-pooling.
        net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        settings = config['pretrained_cfg']
        size, mean, std = settings['input_size'][-1], settings['mean'], settings['std']
        interpolation = settings['interpolation']
        net.load_state_dict(state, strict=True)
    else:
        from transformers import ViTConfig, ViTForImageClassification, SwinConfig, SwinForImageClassification
        expected = 'vit' if architecture == 'vit_b_16' else 'swin'
        if config.get('model_type') != expected:
            raise ValueError('Architecture does not match checkpoint.')
        labels = [config['id2label'][str(i)] for i in range(10)]
        if labels != CIFAR10_CLASSES or len(config['id2label']) != 10:
            raise ValueError('Published checkpoint does not have the CIFAR-10 class order.')
        cls, cfg_cls = ((ViTForImageClassification, ViTConfig) if expected == 'vit'
                        else (SwinForImageClassification, SwinConfig))
        cfg = cfg_cls.from_dict(config)
        cfg._attn_implementation = 'eager'  # Gradients/hooks and ordinary dropout semantics.
        net = cls(cfg)
        net.load_state_dict(state, strict=True)
        prep = json.loads((directory / 'preprocessor_config.json').read_text())
        size = prep['size'] if isinstance(prep['size'], int) else prep['size']['height']
        mean, std = prep['image_mean'], prep['image_std']
        interpolation = {2: 'bilinear', 3: 'bicubic'}[prep['resample']]
        if prep.get('do_center_crop', False):
            raise ValueError('Checkpoint requires unsupported center-crop preprocessing.')
    model = PublishedPixelClassifier(net, size, mean, std, backend, interpolation)
    model.provenance = provenance
    model.mc_dropout_available = any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
                                     and m.p > 0 for m in model.modules())
    return model.to(device).eval()
