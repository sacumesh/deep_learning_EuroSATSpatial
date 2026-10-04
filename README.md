# EuroSAT land-use classification: RGB vs all 13 Sentinel-2 bands
 
Does an ImageNet-pretrained ViT-S/16 classify land use better when it sees all 13 Sentinel-2 bands instead of the 3 RGB bands, everything else being equal?
 
**Short answer: no.** Both inputs reach about 99% test accuracy, and the 13-band model is slightly worse under every training recipe we tried (98.87% vs 99.13% with the tuned recipe, exact McNemar p = 0.13). The pretrained first layer has to be rebuilt for the new bands, while RGB already carries most of the useful shape and texture.
 
## Group members
 
| Name | Email |
|---|---|
| Sachiththa Konara Mudiyanselage | sachiththa.konara-mudiyanselage@edu.dsti.institute, sacume3@gmail.com |
| Naro Couch | naro.kuoch@edu.dsti.institute |
| Yani Lala | yani.lala@edu.dsti.institute |
 
## Repository contents
 
| File | What it is |
|---|---|
| `eurosat_experiments.ipynb` | The full notebook: engine, data, model, evaluation (sections 2 to 8) and the experiments (sections 9 to 11) |
| `EuroSAT_Technical_Project_Summary.pdf` | The project report: main results only, points to the notebook for details |
| `README.md` | This file |
 
## Dataset
 
[EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019): 27,000 Sentinel-2 tiles from 34 European countries, 64 x 64 pixels at 10 m, 10 land-cover classes, all 13 bands as raw reflectance. Loaded with [TorchGeo](https://github.com/microsoft/torchgeo) using its published 60/20/20 split (16,200 / 5,400 / 5,400 tiles), so every number can be reproduced tile for tile.
 
## Model
 
ViT-S/16 (21.7 M parameters) with the `timm` weights `vit_small_patch16_224.augreg_in1k`, pretrained on ImageNet-1k. The output layer is replaced by a new 10-class head. For 13 bands, `timm` widens the first layer by cycling the three pretrained RGB filters over the 13 channels. Tiles are upsampled to 224 x 224 before patching.
 
## Experiments
 
One model and three experiments that build on each other. Every choice is made on the validation split; the test split is only ever scored.
 
```
EXPERIMENTS
│
├── Experiment 1 · before training (sanity check)
│   ├── vit_rgb                 ViT-S/16 + RGB,      untrained   → accuracy near chance?
│   └── vit_all_bands           ViT-S/16 + 13 bands, untrained
│
├── Experiment 2 · the training recipe, on RGB
│   ├── vit_rgb_baseline        default recipe, no augmentation   = trial 0 of the search
│   ├── Optuna search           40 trials over optimizer, learning rate, weight decay, batch size,
│   │                           probe epochs, learning-rate schedule (and momentum for SGD)
│   ├── vit_rgb_tuned           arm A: best recipe, no augmentation
│   └── vit_rgb_tuned_aug       arm B: best recipe + augmentation
│       → augmentation goes into Experiment 3 only if B beats A on the validation split (McNemar p < 0.05)
│
├── Experiment 3 · spectral bands  (the research question)
│   └── vit_all_bands_t20       ViT-S/16 + 13 bands with the recipe chosen in Experiment 2 (trial 20),
│                               compared with the matching RGB arm of Experiment 2 (vit_rgb_tuned)
│
├── Experiment 3b · does the answer depend on the recipe?
│   └── the 5 best recipes of the search, each trained on RGB and on 13 bands
│       (vit_rgb_t23 / vit_all_bands_t23, t38, t37, t22; trial 20 reuses the Experiment 3 pair)
│
└── Extra · earlier runs: the default recipe with augmentation
    ├── vit_rgb                 ViT-S/16 + RGB      + augmentation
    └── vit_all_bands           ViT-S/16 + 13 bands + augmentation
```
 
### Training
 
Every arm is trained by the same `train(model, dataset, config, ...)` call: a few probe epochs with the backbone frozen (only the new head learns), then the whole network is fine-tuned, up to 15 epochs, early stopping on the validation loss, and the weights with the lowest validation loss are kept and scored once on the test split. Experiment 2 tunes the recipe with Optuna (TPE sampler, median pruner); Experiment 3 reuses it unchanged, so its two arms differ only in the input bands.
 
Losses are reported at three levels: the batch loss of every training step, the training loss per epoch (`Σ batch_size · batch_loss / n_train`) and the validation loss (`Σ batch_size · batch_loss / n_val`).
 
### Evaluation
 
Two models are always compared on the same test tiles with the exact McNemar test (p < 0.05). It looks only at the tiles where exactly one model is right and asks whether that split is more lopsided than a coin flip would give. Section 8.1 of the notebook explains the test and checks its assumptions.
 
## Main results
 
| Run | Bands | Recipe | Aug. | Test acc. | Wrong / 5,400 | Macro-F1 |
|---|---|---|---|---|---|---|
| E2 baseline | RGB | default | no | 99.06% | 51 | 0.9904 |
| E2 arm A | RGB | tuned | no | **99.13%** | 47 | 0.9912 |
| E2 arm B | RGB | tuned | yes | 99.09% | 49 | 0.9907 |
| E3 | 13 | tuned | no | 98.87% | 61 | 0.9885 |
| Extra RGB | RGB | default | yes | 99.11% | 48 | 0.9910 |
| Extra 13 | 13 | default | yes | 98.93% | 58 | 0.9889 |
 
| Comparison | Accuracy | Only 1 / only 2 right | p |
|---|---|---|---|
| RGB → 13 bands, tuned recipe (primary test) | 99.13 → 98.87% | 44 / 30 | 0.13 |
| RGB → 13 bands, default recipe + augmentation | 99.11 → 98.93% | 36 / 26 | 0.25 |
| Baseline → tuned recipe (RGB) | 99.06 → 99.13% | 18 / 22 | 0.64 |
| No augmentation → augmentation (RGB, validation) | 99.09 → 99.07% | 25 / 24 | 1.00 |
 
Under all five best recipes of the search (Experiment 3b) the 13-band model was behind RGB by 0.07 to 0.48 points, significantly so for one recipe (p = 0.007). The four vegetation classes, which the infrared bands should help most, lost 37 tiles as a group over the five recipes; SeaLake was the only class that gained consistently.
 
## How to run
 
The notebook was developed on Google Colab (one GPU, about 1.9 GPU-hours in total).
 
```bash
pip install torch torchvision timm torchgeo pytorch-ignite optuna scikit-learn scipy pandas matplotlib seaborn tqdm
```
 
Open `eurosat_experiments.ipynb` and run the cells in order. Sections 2 to 8 define the code, sections 9 to 11 run the experiments. TorchGeo downloads EuroSAT on first use. To reproduce the analysis without training, run the evaluation sections against the saved files in `results/`.
 
Fixed choices for reproducibility: TorchGeo's EuroSAT split, seed 0 for every run and for the Optuna sampler, the hyperparameters in Table 2 of the report. The decision rules (lowest validation loss among trials, augmentation only if McNemar p < 0.05 on validation, one primary test at p < 0.05) were fixed before training.
 
## Libraries
 
`torchgeo` (EuroSAT), `timm` (pretrained backbone), PyTorch, PyTorch Ignite (metrics), Optuna (hyperparameter search), scikit-learn (standardisation and test evaluation), scipy (the exact McNemar test), tqdm, pandas, matplotlib and seaborn (tables, figures and the live training plot).
 
## Use of AI
 
LLMs were used to generate code and to help with the writing of the report. The ideas for the experiments, the choice of libraries, the interpretation of the results, what should be tested, the figures and the overall approach came from us, developed over repeated iterations. When the LLM generated something we did not understand, we asked it to explain it and reviewed the explanation ourselves. References suggested by LLMs were verified before use.
 
## References
 
- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE JSTARS*, 12(7), 2217–2226.
- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*.
- Stewart, A. J., et al. (2022). TorchGeo: Deep learning with geospatial data. *SIGSPATIAL '22*.
- Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD '19*.
- Wightman, R. (2019). PyTorch Image Models (timm). github.com/huggingface/pytorch-image-models
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
 
