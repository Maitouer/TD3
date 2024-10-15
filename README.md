# TD3: Tucker Decomposition Based Dataset Distillation Method for Sequential Recommendation

## Contents

This repository utilizes [PyTorch](https://pytorch.org/) and modern experiment manager tools, [Hydra](https://hydra.cc/) and [Wandb](https://wandb.ai/).

Datasets are downloaded with [Rebole](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj), and preprocessed with [[link]](data#preprocess.ipynb)

Directory structure:

```
.
├── config
│   ├── bert4rec.yaml
│   ├── default.yaml
│   ├── gru4rec.yaml
│   ├── narm.yaml
│   └── sasrec.yaml
├── data
│   ├── preprocess.ipynb
│   ├── processed
│   │   ├── epinions
│   │   │   └── epinions.inter
│   │   ├── magazine
│   │   │   └── magazine.inter
│   │   ├── ml-100k
│   │   │   └── ml-100k.inter
│   │   └── ml-1m
│   │       └── ml-1m.inter
│   └── raw
├── environment.yml
├── README.md
├── script
│   ├── baseline.sh
│   ├── epinions.sh
│   ├── magazine.sh
│   ├── ml100k.sh
│   ├── ml1m.sh
│   └── test.sh
└── src
    ├── baseline.py
    ├── data.py
    ├── distilled_data.py
    ├── evaluator.py
    ├── main.py
    ├── model.py
    ├── pretrainer.py
    ├── trainer.py
    └── utils.py
```

## Run Scripts

1. Clone this repository.
   ```
   $ git clone https://github.com/Maitouer/TD3.git
   $ cd TD3
   ```
2. Prepare environment for **Python 3.10** and install requirements.
   ```
   $ conda env export > environment.yml
   ```
3. Run experiments.
   ```
   $ ./script/magazine.sh
   $ ./script/epinions.sh
   $ ./script/ml100k.sh
   $ ./script/ml1m.sh
   ```
