# Final model checkpoints

The completed study uses three frozen CIFAR-10 classifiers:

- `huggingface/resnet50/`
- `huggingface/vit_b_16/`
- `huggingface/swin_t/`

Each folder retains its model weights, configuration and provenance; transformer folders also contain preprocessing settings. These checkpoints have not been changed by cleanup.

The obsolete RobustBench Standard WideResNet checkpoint was deleted at user request. Archived validation scripts requiring it cannot run without downloading that checkpoint again. The final study does not depend on it. This deletion is local and has not been synced to HPC.
