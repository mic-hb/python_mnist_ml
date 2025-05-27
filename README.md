# MNIST CLI Project: Comprehensive Management Tool

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Current Project Phase (for initial development): Phase 0 - Comprehensive CLI Tool**

This project provides a command-line interface (CLI) for managing the lifecycle of deep learning models for the MNIST handwritten digit recognition task. It leverages TensorFlow 2.x and aims to provide a structured environment for experimentation, training, evaluation, model versioning, robust reporting, and preparation for deployment. The project structure is inspired by `cookiecutter-data-science` and adapted for the specific needs of this ML application.

---

## **Part 1: Project Planning (Developer's Guide)**

This section outlines the step-by-step plan to build the MNIST CLI tool.

### **0. MVP for Interview Showcase (Ultra-Minimal Slice for Today)**

- **Goal:** Demonstrate basic end-to-end ML capability **today**.
- **Tasks (Highest Priority for "Tomorrow"):**
  1. If you haven't, quickly run `cookiecutter https://github.com/drivendataorg/cookiecutter-data-science` to generate the `python_mnist_ml` base.
  2. Setup basic Python environment: `cd python_mnist_ml`, create/activate venv, `pip install tensorflow Pillow typer`.
  3. Create a single Python script in the root of `python_mnist_ml` named `quick_showcase.py`:
     - Defines a simple Keras CNN for MNIST.
     - Loads MNIST data using `tf.keras.datasets.mnist.load_data()`. Preprocess it minimally (normalize, reshape).
     - Trains the model for a few (3-5) epochs.
     - Saves the model using `model.save('showcase_model_sm')` (SavedModel format).
     - Includes a function that loads `'showcase_model_sm'` and predicts on an image path passed as a command-line argument (use `sys.argv`).
  4. Have 1-2 simple digit images ready for prediction (e.g., in `data/02_custom_input_images/`).
- **Showcase:** Briefly explain you used CCDS for structure. Run `python quick_showcase.py path/to/your/digit.png`. Show the (brief) training output and the prediction. Then, _talk about_ the larger, more structured CLI tool and its features using the rest of this document as your guide. Next, _talk about_ the web app and GCP plans using the rest of this document as your guide.

### **1. Initial Project Setup (Leveraging Cookiecutter)**

1. **Generate Project from Template:**
   - `pip install cookiecutter` (if not already installed).
   - `cookiecutter https://github.com/drivendataorg/cookiecutter-data-science`
   - Answer prompts as suggested in the "Adapting Cookiecutter Data Science" section (e.g., `repo_name: python_mnist_ml`, `module_name: src` (or choose another and rename), select `virtualenv`, `requirements.txt`, `flake8+black+isort`, `MIT License`, etc.).
2. **Navigate to Project:** `cd python_mnist_ml`
3. **Adapt CCDS Structure:**
   - Rename `src/your_module_name/` to `src/` if necessary, or adjust plans to use `your_module_name` as the main source package. (The plan below assumes `src/`).
   - Create/Ensure the following top-level directories (if not perfectly matching CCDS output):
     - `configs/hyperparams/`
     - `configs/training_sessions/`
     - `models_trained/` (rename from CCDS `models/`)
     - `models_tfjs/`
     - `data/01_raw/` (CCDS `data/raw/`)
     - `data/02_custom_input_images/`
   - Within `src/`, create our planned subdirectories: `cli/`, `core/`, `utils/`, and `main_cli.py`.
4. **Setup Virtual Environment & Dependencies:**
   - CCDS might set up a basic `requirements.txt`. You'll need to manage this.
   - `python -m venv venv` (if not already done by CCDS setup)
   - `source venv/bin/activate` (or `venv\Scripts\activate`)
   - `pip install -r requirements.txt` (if CCDS provided a useful one)
   - `pip install tensorflow numpy matplotlib Pillow pyyaml typer[all] tqdm scikit-learn tensorflowjs`
   - (Optional But Recommended) `pip install black flake8 isort ipykernel jupyterlab`
   - `pip freeze > requirements.txt` (to update it)
5. **Git Initialization:**
   - CCDS usually initializes a git repository. Make your initial commit.
   - Create `.gitignore` (add `venv/`, `__pycache__/`, `*.egg-info/`, `data/01_raw/mnist.npz` if downloaded manually, `build/`, `dist/`, `*.log`, etc.)
6. **Verify TensorFlow GPU Setup:**
   - Use `notebooks/00_env_check.ipynb` (create this).
   - Run the CUDA check script provided previously. Ensure TensorFlow recognizes the GPU.
7. **Code Formatting & Linting:**
   - Setup `black`, `flake8`, `isort` as per CCDS selection. Configure your IDE or use the `Makefile` targets.
   - Adopt **Black** for code formatting (`pip install black`). Run `black .` before commits.
   - Use **Flake8** for linting (`pip install flake8`). Run `flake8 .`.
   - **Code Formatting Convention:** Adhere to Black.
   - (Optional) Configure pre-commit hooks.
8. **Naming Convention (Code):** Snake case for functions, variables, and filenames (e.g., `my_function.py`). PascalCase for classes (e.g., `MyClass`).

### **2. Directory Structure (Inspired by CCDS & Project Needs)**

```plaintext
python_mnist_ml/
├── .git/
├── .gitignore
├── configs/                                # Project-specific: YAML for hyperparams & training sessions
│   ├── hyperparams/                        # YAML: hp_default.yaml, hp_fast_train.yaml
│   └── training_sessions/                  # YAML: ts_robust_cnn_v1_run1.yaml
├── data/                                   # Data files, organized by processing stage
│   ├── 01_raw/                             # Raw immutable data (e.g., MNIST auto-download via TensorFlow/Keras)
│   ├── 02_custom_input_images/             # User-provided sample images for prediction (testing)
│   ├── 03_tfrecords/                       # (Future) For optimized TFRecord datasets
│   ├── external/                           # Data from third-party sources (CCDS standard)
│   ├── interim/                            # Intermediate data that has been transformed (CCDS standard)
│   └── processed/                          # The final, canonical data sets for modeling (CCDS standard)
├── docs/                                   # Project documentation (e.g., MkDocs if selected)
├── models_trained/                         # Output: Trained models (SavedModel), logs, plots, configs
│   └── {model_name}/                       # Model-specific subdirectory (e.g `mnist_cnn_robust_v1_1.0_20250528_1030_acc0.9910`)
│       ├── saved_model.pb
│       ├── variables/
│       ├── assets/
│       ├── training_config_used.yaml       # Copy of training session config
│       ├── hyperparams_config_used.yaml    # Copy of hyperparams config
│       ├── training_log.csv                # Epoch-wise metrics
│       ├── accuracy_plot.png
│       ├── loss_plot.png
│       └── model_card.md                   # Summary report for this model
├── models_tfjs/                            # Output: TensorFlow.js converted models
├── notebooks/                              # Jupyter notebooks for exploration and experimentation
│   ├── 00_env_check.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_initial_model_training.ipynb
│   └── 03_hyperparam_tuning_yaml.ipynb
├── references/                             # Data dictionaries, manuals, and all other explanatory materials (CCDS standard)
├── reports/                                # Generated analysis and reports (beyond individual model logs) as HTML, PDF, LaTeX, etc. (CCDS standard)
│   └── figures/                            # Central place for generated graphics, plots, and figures to be used in reports
├── requirements.txt                        # Project dependencies
├── setup.py                                # Makes project pip installable (CCDS standard, if module_name was set)
├── src/                                    # Source code for use in this project (or 'your_module_name')
│   ├── __init__.py                         # Makes src a package
│   ├── cli/                                # Typer CLI command modules
│   │   ├── __init__.py
│   │   ├── train_cli.py
│   │   ├── predict_cli.py
│   │   ├── model_cli.py
│   │   └── convert_cli.py
│   ├── core/                               # Core ML logic (data, models, training, etc.)
│   │   ├── __init__.py
│   │   ├── data_processing.py              # Data loading, preprocessing, augmentation
│   │   ├── model_architectures.py          # Model definitions & registry
│   │   ├── training_engine.py              # Training loops, callbacks, saving
│   │   ├── evaluation_engine.py            # Evaluation logic
│   │   ├── prediction_engine.py            # Prediction logic
│   │   └── tfjs_converter_utils.py         # Utilities for TF.js conversion
│   ├── utils/                              # Helper utility functions
│   │   ├── __init__.py
│   │   ├── config_loader.py                # Loads YAML configs
│   │   ├── file_ops.py                     # File/directory operations
│   │   └── logging_setup.py                # Centralized logging configuration
│   └── main_cli.py                         # Main Typer app entry point
├── tests/                                  # Automated tests (unit, integration) for future works
│   ├── __init__.py
│   ├── core/
│   └── cli/
├── Makefile                                # (Optional but Recommended) Makefile with commands like `make data`, `make train` (CCDS standard, extended)
└── README.md                               # This file: the top-level README for developers (project documentation)
```

### **3. Configuration File Design (`configs/`)**

(Same as previous plan: detail structure for hyperparams and training session YAMLs. Implement `src/utils/config_loader.py`.)

1. **Hyperparameter Configs (`configs/hyperparams/<name>.yaml`):**
   - Define sets of hyperparameters. Example: `hp_default.yaml`, `hp_robust_cnn_tuned.yaml`.
   - Structure: `learning_rate`, `batch_size`, `optimizer_name` (e.g., 'adam', 'sgd'), architecture-specific params (e.g., `dropout_conv`, `dense_units`).
2. **Training Session Configs (`configs/training_sessions/<name>.yaml`):**
   - Define a complete training run.
   - Structure:
     - `session_name`: Unique identifier for the run.
     - `model_architecture_name`: Key from `model_architectures.MODEL_REGISTRY`.
     - `model_version_tag`: e.g., "1.0", "1.1-experimental".
     - `hyperparameters_config_file`: Path to a hyperparams YAML (relative to `configs/`).
     - `dataset_name`: e.g., "mnist".
     - `data_preprocessing_options`: `normalize_to_0_1: true`.
     - `data_augmentation_config`: `enabled: true`, `rotation_range: 10`, etc.
     - `training_params`: `epochs: 20`, `validation_split: 0.1`.
     - `callbacks`: `enable_model_checkpoint: true`, `enable_csv_logger: true`, `enable_tensorboard: false`.
     - `notes`: Free text for describing the session.
3. Implement `src/utils/config_loader.py` to load and merge these YAMLs.

### **4. Core Logic Implementation (`src/core/`)**

(Same as previous plan: `data_processing.py`, `model_architectures.py`, `training_engine.py`, `evaluation_engine.py`, `prediction_engine.py`, `tfjs_converter_utils.py`. Emphasize robust image preprocessing in `data_processing.py` and model/log saving conventions in `training_engine.py`.)

1. **`data_processing.py`:**
   - Function `load_mnist_data()`: Returns `(x_train, y_train), (x_test, y_test)`.
   - Function `preprocess_training_data(images, labels, normalize=True)`: Normalizes, reshapes.
   - Function `build_augmentation_pipeline(aug_config)`: Returns a Keras Sequential model of augmentation layers or an `ImageDataGenerator` setup.
   - Function `preprocess_input_image_for_prediction(image_path, target_size=(28,28), invert_colors=False)`: Handles any input image size, resizes with padding, normalizes. Allow `invert_colors` as an option.
2. **`model_architectures.py`:**
   - Implement `MODEL_REGISTRY`, `register_model` decorator, `get_model_architecture()`, `list_available_model_architectures()`.
   - Define at least two model functions (e.g., `build_simple_cnn`, `build_robust_cnn`).
3. **`training_engine.py`:**
   - `run_training_session(training_session_config_path)`:
     - Load session config and associated hyperparams.
     - Prepare data: `load_mnist_data`, `preprocess_training_data`.
     - Setup augmentation if configured.
     - Get model architecture using `model_architectures.get_model_architecture()`.
     - Compile model using optimizer and learning rate from hyperparams.
     - Setup callbacks (`ModelCheckpoint`, `CSVLogger`, custom for plotting/reporting).
       - **Model Naming Convention for `ModelCheckpoint`:**
         `{dataset_name}_{model_arch_name}_{model_version_tag}_{YYYYMMDD}_{HHMM}_acc{val_accuracy:.4f}`
         (e.g., `mnist_CNN_Robust_v1_1.0_20250528_1030_acc0.9910`)
         The output directory for this model will be this name under `models_trained/`.
     - Run `model.fit()`.
     - After fit, save training session config, hyperparams config used, and generate `model_card.md` inside the model's output directory.
     - Generate and save accuracy/loss plots (as PNG) into the model's output directory.
4. **`evaluation_engine.py`:**
   - `evaluate_model(model_path, x_test, y_test)`: Loads SavedModel, runs `model.evaluate()`, returns metrics.
5. **`prediction_engine.py`:**
   - `load_trained_model(model_path)`: Loads SavedModel.
   - `predict_digit(model, preprocessed_image_tensor)`: Returns predicted digit and confidence.
6. **`tfjs_converter_utils.py`:**
   - `convert_saved_model_to_tfjs(saved_model_path, output_tfjs_path)`: Uses `tensorflowjs_converter` CLI via `subprocess`.

### **5. CLI Interface Implementation (`src/cli/` & `src/main_cli.py`)**

(Same as previous plan: modular Typer app with commands for training, model management, prediction, and TF.js conversion.)

1. **`main_cli.py`:** Setup main Typer app and add sub-apps.
2. **`train_cli.py`:**
   - `train start --config <path_to_training_session_yaml>`: Calls `training_engine.run_training_session`.
3. **`model_cli.py`:**
   - `model list-arch`: Calls `model_architectures.list_available_model_architectures`.
   - `model list-trained [--sort-by <accuracy|date>]`: Scans `models_trained/`, parses names, displays info.
   - `model info <trained_model_name_or_path>`: Displays `model_card.md` content, metrics from CSV, path to plots.
   - `model evaluate <trained_model_path>`: Calls `evaluation_engine.evaluate_model`.
4. **`predict_cli.py`:**
   - `predict image --model <trained_model_path>`: Prompts for image path, calls `data_processing.preprocess_input_image_for_prediction`, then `prediction_engine.predict_digit`.
5. **`convert_cli.py`:**
   - `convert tfjs --model <trained_model_path> --output <output_dir_for_tfjs_model>`: Calls `tfjs_converter_utils.convert_saved_model_to_tfjs`.
6. **Utilities (`src/utils/`)**:
   - `logging_setup.py`: Basic logging config.
   - `file_ops.py`: Helpers for scanning directories, ensuring paths exist.

### **6. Robust Reporting Feature (Integrated into `training_engine` and `model_cli`)**

(Same as previous plan: generation of plots, CSV logs, config copies, and `model_card.md` within each `models_trained/<model_name>/` directory.)

1. **During Training (`training_engine.py`):**

   - Save copies of the exact `training_session.yaml` and `hyperparameters.yaml` used into the specific trained model's output directory (e.g., `models_trained/mnist_..._accX.XXXX/`).
   - Ensure `CSVLogger` saves epoch-by-epoch `loss`, `accuracy`, `val_loss`, `val_accuracy`.
   - Generate and save `accuracy_plot.png` and `loss_plot.png`.
   - **`model_card.md` Generation:** After training, create a markdown file in the model's directory:

     ```markdown
     # Model Card: {{model_full_name}}

     - **Architecture:** {{model_architecture_name}} (Version: {{model_version_tag}})
     - **Trained At:** {{timestamp}}
     - **Training Duration:** {{duration}}
     - **Base Dataset:** {{dataset_name}}
     - **Training Session Config:** `training_config_used.yaml`
     - **Hyperparameters Config:** `hyperparams_config_used.yaml`

     ## Final Metrics

     - **Training Loss:** {{final_train_loss}}
     - **Training Accuracy:** {{final_train_accuracy}}
     - **Validation Loss:** {{final_val_loss}}
     - **Validation Accuracy:** {{final_val_accuracy}}
     - (Test metrics if evaluation run)

     ## Training History

     - [Training Log CSV](./training_log.csv)
     - Accuracy Plot: ![Accuracy](./accuracy_plot.png)
     - Loss Plot: ![Loss](./loss_plot.png)

     ## Notes

     {{notes_from_training_session_config}}
     ```

2. **CLI Display (`model_cli.py model info ...`):**

   - Read and pretty-print the `model_card.md`.
   - Optionally, re-plot graphs if needed or just point to the saved PNGs.

### **7. Makefile Enhancement (Optional but Recommended for Automation)**

- Review the `Makefile` generated by CCDS.
- Add/modify targets for:

  - `lint`: `flake8 src tests`
  - `format`: `black src tests`
  - `install_dev`: Installs `requirements.txt` plus dev tools like `pytest`.
  - `clean`: Removes `__pycache__`, `*.pyc`, build artifacts.
  - `train_default_session`: Runs `python src/main_cli.py train start --config configs/training_sessions/ts_default_run.yaml`
  - `list_models`: Runs `python src/main_cli.py model list-trained`
  - `convert_default_model`: Converts a specific (or latest) model to TF.js.
  - (Future) `test`: Runs `pytest`.

- Example:
- Create `Makefile` in project root:

```makefile
.PHONY: help install setup_notebook lint format clean_pyc clean_build train_default predict_example convert_default

help:
  @echo "Commands:"
  @echo "  install         : Install dependencies from requirements.txt"
  @echo "  setup_notebook  : Install Jupyter kernel for this project"
  @echo "  lint            : Lint code with flake8"
  @echo "  format          : Format code with black"
  @echo "  clean_pyc       : Remove Python file artifacts"
  @echo "  clean_build     : Remove build artifacts"
  @echo "  train_default   : Train with a default training session config"
  @echo "  predict_example : Predict on a sample image with a default model"
  @echo "  convert_default : Convert a default trained model to TF.js"

install:
  pip install -r requirements.txt

setup_notebook:
  pip install ipykernel
  python -m ipykernel install --user --name=python_mnist_ml --display-name="Python (MNIST ML)"

lint:
  flake8 src tests

format:
  black src tests

# ... other clean targets ...

train_default:
  python src/main_cli.py train start --config configs/training_sessions/ts_default_session.yaml

# ... other example targets ...
```

### **8. Testing Strategy (Future)**

- Plan to use `pytest`.
- Write unit tests for functions in `src/core/` (especially data processing, config loading).
- Write integration tests for CLI commands using `typer.testing.CliRunner`. Place these in the `tests/` directory.
- Plan for unit tests (`pytest`) for core functions in `data_processing.py`, `model_architectures.py`.
- Plan for integration tests for CLI commands (e.g., using `typer.testing.CliRunner`).

---

## **Part 2: Project Documentation (User & Contributor Guide)**

(This section assumes the CLI tool as planned above is fully implemented.)

### **1. Project Overview**

**MNIST CLI Project** is a command-line utility designed to streamline the lifecycle of developing, training, evaluating, and managing deep learning models for the MNIST handwritten digit recognition task. Built using TensorFlow 2.x and structured with best practices inspired by `cookiecutter-data-science`, it offers a robust environment for reproducible ML experimentation.

This tool is built with flexibility in mind, allowing users to define custom model architectures, configure training sessions through simple YAML files, and manage multiple trained models with detailed reporting.

### **2. Key Information**

- **Primary Task:** Handwritten Digit Recognition (0-9)
- **Dataset:** MNIST (Modified National Institute of Standards and Technology database)
- **Core Technology:** Deep Learning (Convolutional Neural Networks)
- **Main Library:** TensorFlow 2.x (with Keras API)
- **CLI Framework:** Typer
- **Configuration:** YAML
- **Code Style:** Black, Flake8, isort
- **Environment Management:** virtualenv (with `requirements.txt`)
- **Key Concepts:** Supervised Learning, Image Classification, CNNs, Hyperparameter Tuning, Model Versioning, Model Deployment (TF.js conversion).

### **3. Features**

(Same list as in the previous README plan: Experimentation, Training, Model Management, Evaluation, Prediction, Deployment Preparation, Reporting.)

- **Experimentation:**
  - Support for multiple, user-defined CNN architectures.
  - Flexible hyperparameter configuration via YAML files.
  - Configurable data preprocessing and augmentation for training.
- **Training:**
  - Start training sessions using comprehensive YAML configuration files.
  - Automatic logging of epoch-wise metrics (loss, accuracy) to CSV.
  - Generation of accuracy and loss plots for each training run.
- **Model Management:**
  - List available model architecture definitions.
  - List all trained models with key metadata (name, version, date, accuracy).
  - Detailed information for each trained model, including:
    - Configuration files used (training session, hyperparameters).
    - Complete training history logs and plots.
    - A "Model Card" summarizing the model.
  - Standardized naming convention for trained models.
  - Models saved in TensorFlow's SavedModel format.
- **Evaluation:**
  - Evaluate trained models on the MNIST test set.
- **Prediction:**
  - Predict digits from local image files (supports various image sizes with automatic preprocessing).
- **Deployment Preparation:**
  - Convert trained models (SavedModel) to TensorFlow.js format for web deployment.
- **Reporting:**
  - Automatic generation of model cards for each training run.
  - Visual plots for training progress.

### **4. Additional (Add-on) Features (Future Roadmap)**

(Same list as in the previous README plan.)

- Integration with experiment tracking tools (e.g., MLflow, Weights & Biases).
- Support for other image datasets with minor configuration changes.
- Automated hyperparameter optimization (e.g., using KerasTuner).
- More advanced data augmentation techniques.
- Web UI for interacting with the CLI's capabilities (Phase 1 of the larger project).
- Automated testing suite (unit and integration tests).
- Support for distributed training (for larger models/datasets).

### **5. Project Structure (Derived from Cookiecutter Data Science)**

```plaintext
python_mnist_ml/
├── .git/
├── .gitignore
├── configs/                                # Project-specific: YAML for hyperparams & training sessions
│   ├── hyperparams/                        # YAML: hp_default.yaml, hp_fast_train.yaml
│   └── training_sessions/                  # YAML: ts_robust_cnn_v1_run1.yaml
├── data/                                   # Data files, organized by processing stage
│   ├── 01_raw/                             # Raw immutable data (e.g., MNIST auto-download via TensorFlow/Keras)
│   ├── 02_custom_input_images/             # User-provided sample images for prediction (testing)
│   ├── 03_tfrecords/                       # (Future) For optimized TFRecord datasets
│   ├── external/                           # Data from third-party sources (CCDS standard)
│   ├── interim/                            # Intermediate data that has been transformed (CCDS standard)
│   └── processed/                          # The final, canonical data sets for modeling (CCDS standard)
├── docs/                                   # Project documentation (e.g., MkDocs if selected)
├── models_trained/                         # Output: Trained models (SavedModel), logs, plots, configs
│   └── {model_name}/                       # Model-specific subdirectory (e.g `mnist_cnn_robust_v1_1.0_20250528_1030_acc0.9910`)
│       ├── saved_model.pb
│       ├── variables/
│       ├── assets/
│       ├── training_config_used.yaml       # Copy of training session config
│       ├── hyperparams_config_used.yaml    # Copy of hyperparams config
│       ├── training_log.csv                # Epoch-wise metrics
│       ├── accuracy_plot.png
│       ├── loss_plot.png
│       └── model_card.md                   # Summary report for this model
├── models_tfjs/                            # Output: TensorFlow.js converted models
├── notebooks/                              # Jupyter notebooks for exploration and experimentation
│   ├── 00_env_check.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_initial_model_training.ipynb
│   └── 03_hyperparam_tuning_yaml.ipynb
├── references/                             # Data dictionaries, manuals, and all other explanatory materials (CCDS standard)
├── reports/                                # Generated analysis and reports (beyond individual model logs) as HTML, PDF, LaTeX, etc. (CCDS standard)
│   └── figures/                            # Central place for generated graphics, plots, and figures to be used in reports
├── requirements.txt                        # Project dependencies
├── setup.py                                # Makes project pip installable (CCDS standard, if module_name was set)
├── src/                                    # Source code for use in this project (or 'your_module_name')
│   ├── __init__.py                         # Makes src a package
│   ├── cli/                                # Typer CLI command modules
│   │   ├── __init__.py
│   │   ├── train_cli.py
│   │   ├── predict_cli.py
│   │   ├── model_cli.py
│   │   └── convert_cli.py
│   ├── core/                               # Core ML logic (data, models, training, etc.)
│   │   ├── __init__.py
│   │   ├── data_processing.py              # Data loading, preprocessing, augmentation
│   │   ├── model_architectures.py          # Model definitions & registry
│   │   ├── training_engine.py              # Training loops, callbacks, saving
│   │   ├── evaluation_engine.py            # Evaluation logic
│   │   ├── prediction_engine.py            # Prediction logic
│   │   └── tfjs_converter_utils.py         # Utilities for TF.js conversion
│   ├── utils/                              # Helper utility functions
│   │   ├── __init__.py
│   │   ├── config_loader.py                # Loads YAML configs
│   │   ├── file_ops.py                     # File/directory operations
│   │   └── logging_setup.py                # Centralized logging configuration
│   └── main_cli.py                         # Main Typer app entry point
├── tests/                                  # Automated tests (unit, integration) for future works
│   ├── __init__.py
│   ├── core/
│   └── cli/
├── Makefile                                # (Optional but Recommended) Makefile with commands like `make data`, `make train` (CCDS standard, extended)
└── README.md                               # This file: the top-level README for developers (project documentation)
```

**Key Directory Explanations:** (Briefly explain the purpose of main directories, highlighting those from CCDS and our specific additions like `configs/`, `models_trained/`, `src/cli/`, etc.)

**Key Directories:**

- `configs/`: Contains YAML configuration files for hyperparameter sets and training sessions.
- `data/`: Stores raw datasets and user-provided images.
- `models_trained/`: Output directory for all trained models, each in its own versioned subdirectory containing the SavedModel, logs, plots, and configs.
- `models_tfjs/`: Output directory for TensorFlow.js converted models.
- `notebooks/`: Jupyter notebooks for exploration, experimentation, and initial development.
- `reports/`: (If used for global reports) Centralized reports or summary figures.
- `src/`: All source code.
  - `src/cli/`: Modules defining the Typer CLI commands.
  - `src/core/`: Core logic for data processing, model building, training, prediction, etc.
  - `src/utils/`: Utility functions for configuration loading, file operations, logging.
  - `src/main_cli.py`: The main entry point for the CLI application.
- `tests/`: (Future) Automated tests.

### **6. Configuration Files Guide**

(Same as the previous README plan: Explain `hyperparams/*.yaml` and `training_sessions/*.yaml` with examples.)

This project uses YAML files for managing configurations, promoting reproducibility and easy experimentation.

- **Hyperparameter Configurations (`configs/hyperparams/`):**

  - These files define sets of hyperparameters that can be reused across different training sessions.
  - **Example (`hp_default.yaml`):**
    ```yaml
    # configs/hyperparams/hp_default.yaml
    learning_rate: 0.001
    batch_size: 128
    optimizer_name: "adam" # Supported: 'adam', 'sgd', etc. (as per training_engine)
    # Model-specific hyperparams (ensure your model architecture function uses these)
    dropout_rate_conv: 0.25
    dropout_rate_dense: 0.5
    cnn_block1_filters: 32
    ```
  - To create a new set, copy an existing file, rename it (e.g., `hp_aggressive_lr.yaml`), and modify the values.

- **Training Session Configurations (`configs/training_sessions/`):**

  - These files define all aspects of a specific training run, linking together a model architecture, hyperparameters, data settings, and training parameters.
  - **Example (`ts_my_first_run.yaml`):**

    ```yaml
    # configs/training_sessions/ts_my_first_run.yaml
    session_name: "MyFirstMNISTRun" # Used for unique output directory if not overridden by CLI
    model_architecture_name: "CNN_MNIST_Robust_v1" # Must match a key in src.core.model_architectures.MODEL_REGISTRY
    model_version_tag: "1.0-alpha" # Your custom version tag for this setup
    hyperparameters_config_file: "hyperparams/hp_default.yaml" # Path relative to configs/

    dataset_name: "mnist"
    data_preprocessing_options:
      normalize_to_0_1: true
    data_augmentation_config:
      enabled: true
      rotation_range: 10
      width_shift_range: 0.05
      height_shift_range: 0.05
      zoom_range: 0.05

    training_params:
      epochs: 15
      validation_split: 0.1 # Fraction of training data to use for validation

    callbacks:
      enable_model_checkpoint: true # Saves the best model during training
      enable_csv_logger: true # Logs epoch metrics to CSV
      enable_tensorboard: false # Enable if you want to use TensorBoard

    output_base_dir: "../models_trained" # Default base directory for output
    notes: "Initial experimental run with robust CNN and default hyperparameters."
    ```

  - To create a new training session, copy an existing file, rename it, and customize its parameters. Reference the desired hyperparameter file.

### **7. How the Model Works (MNIST CNN Overview)**

The core models used in this project are Convolutional Neural Networks (CNNs), which are particularly well-suited for image recognition tasks.

1. **Input:** The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). Before being fed to the network, these images are preprocessed:

   - **Normalization:** Pixel values (typically 0-255) are scaled to a smaller range (e.g., 0-1). This helps with training stability and convergence.
   - **Reshaping:** Images are reshaped to include a channel dimension (e.g., `(28, 28, 1)` for grayscale).
   - For custom input images during prediction, the tool provides robust preprocessing to convert images of any size to this required format, including grayscale conversion, aspect-ratio preserving resize with padding, and normalization.

2. **Convolutional Layers (`Conv2D`):**

   - These layers apply a set of learnable filters (kernels) to the input image. Each filter detects specific features like edges, corners, or textures.
   - The output is a set of "feature maps" highlighting these detected features.

3. **Activation Functions (e.g., `ReLU` - Rectified Linear Unit):**

   - Applied after convolutional layers to introduce non-linearity, allowing the network to learn more complex patterns. ReLU sets all negative values to zero.

4. **Batch Normalization (`BatchNormalization`):**

   - Normalizes the output of the previous layer during training. This helps stabilize learning, accelerates training, and can act as a regularizer.

5. **Pooling Layers (`MaxPooling2D`):**

   - Reduce the spatial dimensions (width and height) of the feature maps, making the network more robust to variations in feature positions and reducing computational load. Max pooling takes the maximum value from a small window of the feature map.

6. **Dropout Layers (`Dropout`):**

   - A regularization technique to prevent overfitting. During training, it randomly sets a fraction of input units to 0 at each update, forcing the network to learn more robust features.

7. **Flatten Layer (`Flatten`):**

   - Converts the 2D feature maps from the convolutional/pooling blocks into a 1D vector, preparing it for the fully connected layers.

8. **Dense Layers (Fully Connected Layers):**

   - Standard neural network layers where each neuron is connected to all neurons in the previous layer. These layers learn higher-level combinations of features.
   - The final dense layer has a number of neurons equal to the number of classes (10 for MNIST) and uses a `softmax` activation function.

9. **Softmax Activation (Output Layer):**

   - Converts the raw output scores (logits) from the final dense layer into a probability distribution over the classes. Each output neuron will have a value between 0 and 1, and all values will sum to 1, representing the model's confidence for each digit.

10. **Training (Optimization):**

    - The model learns by comparing its predictions to the true labels using a **loss function** (e.g., `sparse_categorical_crossentropy` for integer labels).
    - An **optimizer** (e.g., `Adam`, `SGD`) adjusts the model's weights (filters in Conv2D, weights in Dense layers) to minimize this loss function through a process called backpropagation.

### **8. Quickstart Guide**

(Same as the previous README plan, ensure paths reflect the CCDS-based structure if needed, e.g., `make install_dev` might be a new first step from the Makefile.)

1. **Clone the Repository (if applicable) or Setup Project:**

   ```bash
   # Assuming you have the project files in python_mnist_ml/
   cd python_mnist_ml
   ```

2. **Setup Environment & Install Dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Train a Model with a Default Configuration:**

   - First, ensure a default training session config exists, e.g., `configs/training_sessions/ts_default_run.yaml`.
   - This default config should point to a default hyperparameter config, e.g., `configs/hyperparams/hp_default.yaml`.

   ```bash
   python src/main_cli.py train start --config configs/training_sessions/ts_default_run.yaml
   ```

   - Note the full name of the trained model directory that gets created under `models_trained/`. It will look something like `mnist_CNN_Simple_v1_1.0_YYYYMMDD_HHMM_acc0.XXXX`.

4. **List Trained Models:**

   ```bash
   python src/main_cli.py model list-trained
   ```

5. **Predict a Digit from an Image:**

   - You'll need a sample digit image. You can create one or find one online. Save it to `data/02_custom_input_images/my_digit.png`.
   - Replace `<YOUR_TRAINED_MODEL_NAME>` with the directory name from step 3 or `model list-trained`.

   ```bash
   python src/main_cli.py predict image --model models_trained/<YOUR_TRAINED_MODEL_NAME>
   # The CLI will then prompt you for the image path:
   # Enter the path to your image file: data/02_custom_input_images/my_digit.png
   ```

### **9. How-To Guides**

(Same as the previous README plan: Detailed usage for each CLI command group.)

**(This section would contain detailed instructions for each command group and feature)**

- **How to Define a New Model Architecture:**

  1. Open `src/core/model_architectures.py`.
  2. Define a new Python function that returns a `tf.keras.Model` (e.g., `build_my_custom_cnn(...)`).
  3. Decorate it with `@register_model("MyCustomCNN_v1")`.
  4. You can now reference `"MyCustomCNN_v1"` in your training session YAML files.

- **How to Create New Hyperparameter & Training Session Configs:**

  1. Navigate to `configs/hyperparams/` or `configs/training_sessions/`.
  2. Copy an existing `.yaml` file (e.g., `cp hp_default.yaml hp_experimental.yaml`).
  3. Edit the new YAML file with your desired parameters.
  4. Reference the new hyperparameter file in your new training session config if needed.
  5. Use the new training session config with `python src/main_cli.py train start --config ...`.

- **How to View Information about a Trained Model:**

  ```bash
  python src/main_cli.py model info <trained_model_name_or_path>
  ```

  This will display the model card, including links to its specific configuration files, logs, and plots.

- **How to Evaluate a Trained Model:**

  ```bash
  python src/main_cli.py model evaluate <trained_model_path>
  ```

- **How to Convert a Trained Model to TensorFlow.js Format:**

  ```bash
  python src/main_cli.py convert tfjs --model models_trained/<YOUR_TRAINED_MODEL_NAME> --output models_tfjs/<YOUR_TFJS_MODEL_NAME>
  ```

- **How to Interpret Training Logs and Plots:**

  - The `training_log.csv` file in each model's directory contains epoch-by-epoch metrics: `epoch`, `accuracy`, `loss`, `val_accuracy`, `val_loss`.
  - `accuracy_plot.png`: Shows training and validation accuracy. Ideally, both should increase and converge. A large gap might indicate overfitting.
  - `loss_plot.png`: Shows training and validation loss. Ideally, both should decrease and converge. Increasing validation loss while training loss decreases is a strong sign of overfitting.

### **10. Naming Conventions**

(Same as the previous README plan: For model architectures, config files, trained models, TF.js models. Add conventions for branches if using Git extensively, e.g., `feature/`, `bugfix/`, `release/`.)

- **Model Architecture Names (in `model_architectures.py` registry):** `Dataset_Purpose_ArchType_Version` (e.g., `MNIST_Classifier_CNN_Simple_v1`, `MNIST_Classifier_CNN_Robust_v1.1`).
- **Hyperparameter Config Files (`configs/hyperparams/`):** `hp_<descriptive_name>.yaml` (e.g., `hp_default.yaml`, `hp_high_lr_sgd.yaml`).
- **Training Session Config Files (`configs/training_sessions/`):** `ts_<descriptive_name>.yaml` (e.g., `ts_robust_cnn_initial_run.yaml`).
- **Trained Model Directories (`models_trained/`):**
  `{dataset_name}_{model_arch_name_from_config}_{model_version_tag_from_config}_{YYYYMMDD}_{HHMM}_acc{val_accuracy:.4f}`
  (e.g., `mnist_CNN_Robust_v1_1.0-alpha_20250528_1430_acc0.9921`).
- **TensorFlow.js Model Directories (`models_tfjs/`):** Typically derived from the trained model name (e.g., `mnist_CNN_Robust_v1_1.0-alpha_20250528_1430_acc0.9921_tfjs`).

### **11. Code Style & Formatting**

- **Python Code:** Formatted using **Black** (see `pyproject.toml` if CCDS sets it up, or run `black .`).
- **Linting:** Checked with **Flake8** (see configuration if CCDS sets it up, or run `flake8 .`).
- **Import Sorting:** Managed by **isort** (see configuration if CCDS sets it up, or run `isort .`).
- **Docstrings:** Use Google-style Python docstrings.
- **Type Hinting:** Use type hints for function signatures and variables where beneficial for clarity.

### **12. Makefile Usage**

- The `Makefile` provides shortcuts for common development tasks.
- Run `make help` to see available commands.
- **Common commands:**
  - `make install_dev` (or similar from CCDS for setting up env and dev tools)
  - `make lint`
  - `make format`
  - `make train_default_session` (our custom target)
  - (Future) `make test`

### **13. Troubleshooting / FAQ**

- **Q: TensorFlow not finding GPU?**
  - A: Ensure NVIDIA drivers, CUDA Toolkit, and cuDNN are correctly installed and their versions are compatible with your TensorFlow version. Run the `00_env_check.ipynb` notebook. Make sure you installed `tensorflow` and not `tensorflow-cpu`.
- **Q: `tensorflowjs_converter` command not found?**
  - A: Ensure you have installed `tensorflowjs`: `pip install tensorflowjs`. Make sure your virtual environment's `Scripts` (Windows) or `bin` (Linux/macOS) directory is in your system PATH, or call the converter using its full path or `python -m tensorflowjs.converters.converter_cli ...`.
- **Q: Low accuracy on custom input images for prediction?**
  - A: Check the preprocessing steps in `src/core/data_processing.py preprocess_input_image_for_prediction`. It must closely match the preprocessing applied to the MNIST training data. Pay attention to:
    - Grayscale conversion.
    - Color inversion (MNIST is typically white digit on black background. If your image is black on white, it might need inversion).
    - Normalization range (0-1 or -1 to 1, etc.).
    - Resizing and padding method.

### **14. Contributing**

(Standard contribution guidelines: fork, branch, code style, tests, PR.)
(Details on how to contribute if this were an open project: code style, testing requirements, pull request process.)

### **15. License**

MIT License
