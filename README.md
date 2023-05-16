# Unified Tracking in PyTorch

This package is a robust object tracking framework for PyTorch. It facilitates multi-stage and cascaded tracking algorithms under various modular configurations and assignment algorithms. This open-source implementation is designed to facilitate research in computer vision and machine learning. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [Citation](#citation)
- [License](#license)
- [Recommendations](#recommendations)

## Installation

Ensure your environment meets the following requirements:

- `python >= 3.9`
- `torch >= 2.0`

Install via PyPI using the following command:

```bash
pip install unitrack
```

## Usage

The following example demonstrates object tracking across a sequence with detections that have `category` and `position` fields. This script tracks objects, updates internal state buffers for each frame, and prints the assigned IDs.

```python3
import unitrack

frames = [
    {
        "category": torch.ones(1 + frame * 2, dtype=torch.long),
        "position": (torch.arange(1 + frame * 2, dtype=dtype)).unsqueeze(1),
    }
    for frame in range(0, 10)
]

tracker = unitrack.MultiStageTracker(
    fields={
        "pos": unitrack.fields.Value(key="pos_key"),
        "categories": unitrack.fields.Value(key="pred_class"),
    },
    stages=[unitrack.stages.Association(cost=costs.Distance("pos"), assignment=unitrack.assignment.Jonker(10))],
)

memory = TrackletMemory(
    states={
        "pos": unitrack.states.Value(dtype),
        "categories": unitrack.states.Value(dtype=torch.long),
    }
)

for frame, detections in enumerate(frames):
    ctx = unitrack.Context(None, detections, frame=frame)
    obs = memory.observe()
    obs, new = tracker(ctx, obs)
    ids = tracks.update(ctx, obs, new)

    print(f"Assigned tracklet IDs {ids.tolist()} @ frame {frame}")
```

## Documentation

Technical documentation is provided inline with the source code.

## Contribution

Contributions that maintain backwards compatibility are welcome.

## Citation

If you utilize this package in your research, please cite the following paper:

```bib
@article{unifiedperception2023,
    title={Unified Perception: Efficient Depth-Aware Video Panoptic Segmentation with Minimal Annotation Costs},
    author={Kurt Stolle and Gijs Dubbelman},
    journal={arXiv preprint arXiv:2303.01991},
    year={2023}
}
```

Access the full paper [here](https://arxiv.org/abs/2303.01991).

## License

This project is licensed under [MIT License](LICENSE).

## Recommendations

The contents of this repository are designed for research purposes and is not recommended for use in production environments. It has not undergone testing for scalability or stability in a commercial context. Please use this tool within its intended scope.


