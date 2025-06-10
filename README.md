# starmap

## Starmap for ASTRON:
starmap4astron/
├── models/                           # pure model definitions
│   ├── __init__.py                   # re‑exports Encoder, Decoder, LC2Img
│   ├── encoder.py                    # ResBlock1D, LightcurveEncoder1D (1D→latent grid)
│   ├── decoder.py                    # Decoder2D (latent grid → image)
│   └── lc2img.py                     # LC2Img wrapper combining encoder+decoder
│
├── trainers/                         # training logic (PyTorch Lightning)
│   └── lc2img_module.py              # LightningModule: builds LC2Img, defines forward, training_step, configure_optimizers
│
├── data/
│   └── dataset.py                    # StarryNPZDataset: loads `.npz` of (flux, image) pairs into (1×N_PT → 1×H×W) tensors
│
├── tests/                            # unit tests
│   ├── test_encoder.py               # shape/grad tests for ResBlock1D & LightcurveEncoder1D
│   └── test_decoder.py               # shape tests for Decoder2D (if you added)
│
├── train.py                          # CLI entry‑point: parses args, instantiates LC2ImgModule, DataLoader, Trainer
├── inference.py                      # standalone inference script (loads checkpoint, runs one LC→image, saves PNG)
├── run_training.sh                   # bash wrapper: activates conda, runs train.py, finds checkpoint, calls inference.py
└── requirements.txt or environment.yml (optional)


Project about reconstructing a star image from viewed intensities during occultations.


hatp11keplersrc.csv:
Kepler light curve of HAT-P-11 - columns are time (Barycentric julian date - 2454833) and flux (median normalized to 1).
