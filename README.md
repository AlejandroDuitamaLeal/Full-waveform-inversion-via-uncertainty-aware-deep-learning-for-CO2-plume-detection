# Seismic Inversion FWI — FPN

This repository contains three notebooks for training, calibrating, and
evaluating a probabilistic FPN model for seismic inversion on the
Kimberlina CO2 dataset.

## `1-TrainModel.ipynb` — Model Training

- Loads the trace paths (input) and velocity model paths (output) from
  the Kimberlina CO2 train/validation folders.
- Splits the training data into training, test, and calibration subsets
  (using shuffled indices, keeping trace-model pairs aligned), while the
  validation set is left unchanged.
- Computes the global min/max velocity from the training labels.
- Defines a `SeismicSequence` (`tf.keras.utils.Sequence`) that loads
  `.npz` trace/velocity pairs, resizes the velocity labels, and
  normalizes them using the global min/max.
- Builds a model consisting of convolution/pooling/resizing layers
  followed by an FPN with a SE-ResNet50 backbone (`segmentation_models`).
- Compiles the model with a negative log-likelihood loss function
  (`-y_pred.log_prob(y_true)`) and metrics (MAE, RMSE).
- Trains the model using `ReduceLROnPlateau`, `EarlyStopping`, and
  `ModelCheckpoint` callbacks.

## `2-Calibration.ipynb` — Pixel-Wise Calibration

- Rebuilds the same data pipeline and model architecture as in
  `1-TrainModel.ipynb`.
- Loads the trained model weights and the saved training history.
- Plots the training/validation loss and metric curves.
- Runs the model on the validation set multiple times per batch to
  obtain the mean prediction, the epistemic uncertainty (variance across
  runs), the aleatoric uncertainty (mean of predicted variances), and
  the total standard deviation.
- Uses `uncertainty_toolbox` to compute calibration metrics (MACE,
  RMSCE, miscalibration area) for individual pixels, and shows the
  effect of recalibrating a single pixel.
- Performs full-image recalibration: for every pixel, computes the
  optimal scalar ratio that minimizes the miscalibration area
  (parallelized with `ProcessPoolExecutor`), and compares miscalibration
  before and after recalibration.
- Saves the resulting per-pixel recalibration ratio map, the
  before/after miscalibration maps, the recalibrated standard
  deviations, and a metadata JSON file describing how they were
  generated.

## `3-Results.ipynb` — Results

- Rebuilds the same data pipeline and model architecture as in the
  previous notebooks and loads the trained model weights.
- Loads the saved recalibration ratio map and its metadata.
- Generates predictions on the validation set with the trained model.
- Runs the model repeatedly on a selected batch to compute the mean
  prediction, epistemic uncertainty, aleatoric uncertainty, and total
  standard deviation.
- De-normalizes the predicted and reference velocity maps and the
  uncertainty maps using the known velocity range, and resizes them to
  the original grid size.
- Applies the saved recalibration ratio map to the aleatoric and
  epistemic uncertainties.
- Plots the reference velocity, predicted velocity, and the
  aleatoric/epistemic uncertainties (before and after recalibration) as
  a combined figure.
- Computes R², MAE, and MSE (and optionally IoU) between reference and
  predicted velocity for a selected example, and generates map and
  scatter-plot comparisons.
