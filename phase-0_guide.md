# MNIST Digit Recognizer: CLI Management Tool

**Current Project Phase (for initial development): Phase 0 - Comprehensive CLI Tool**

This document outlines the planning and documentation for the MNIST Digit Recognizer CLI Management Tool. It is designed to be a comprehensive utility for training, evaluating, managing, and deploying MNIST models.

---

## **Part 1: Project Planning (Developer's Guide)**

This section details the step-by-step plan to build the MNIST CLI tool.

### **0. MVP for Interview Showcase (Ultra-Minimal Slice)**

- **Goal:** Demonstrate basic end-to-end ML capability.
- **Tasks (Highest Priority for "Tomorrow"):**
  1.  Setup basic Python environment with TensorFlow.
  2.  Write a single Python script (`quick_train_predict.py`):
      - Defines a simple CNN for MNIST.
      - Loads MNIST data.
      - Trains the model for a few epochs.
      - Saves the model (SavedModel format).
      - Includes a function to load the saved model and predict on a sample image path given as a command-line argument.
  3.  Prepare 1-2 sample digit images for prediction.
- **Showcase:** Run the script, show training (briefly), then predict on your sample images. Then, _talk about_ the larger CLI tool, web app, and GCP plans using the rest of this document as your guide.

### **1. Project Setup & Environment**

1.  **Create Project Root Directory:** `python_mnist_ml`
2.  **Initialize Git:** `git init`
3.  **Setup Python Virtual Environment:**
    - `python -m venv venv`
    - `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
    - Create `.gitignore` (add `venv/`, `__pycache__/`, `*.egg-info/`, `data/01_raw/mnist.npz` if downloaded manually, `build/`, `dist/`, `*.log`, etc.)
4.  **Install Core Libraries:**
    - `pip install tensorflow numpy matplotlib Pillow pyyaml typer[all] tqdm scikit-learn`
    - (Optional but Recommended) `pip install black flake8 ipykernel jupyterlab` (for notebooks and formatting)
    - `pip freeze > requirements.txt`
5.  **Verify TensorFlow GPU Setup (in a Jupyter Notebook or test script):**
    - Run the CUDA check script provided previously. Ensure TensorFlow recognizes the GPU.
6.  **Code Formatting (Convention):**
    - Adopt **Black** for code formatting (`pip install black`). Run `black .` before commits.
    - Use **Flake8** for linting (`pip install flake8`). Run `flake8 .`.
    - (Optional) Configure pre-commit hooks.

### **2. Directory Structure (Inspired by CCDS & Project Needs)**

```plaintext
python_mnist_ml/
├── .git/
├── .gitignore
├── configs/
│   ├── hyperparams/                                      # YAML: hp_default.yaml, hp_fast_train.yaml
│   └── training_sessions/                                # YAML: ts_robust_cnn_v1_run1.yaml
├── data/
│   ├── 01_raw/                                           # MNIST data auto-downloads here via TensorFlow/Keras
│   ├── 02_custom_input_images/                           # Sample images for prediction testing
│   └── 03_tfrecords/                                     # (Future) For optimized TFRecord datasets
├── docs/                                                 # For additional documentation (e.g., architecture diagrams)
├── models_trained/                                       # Output: Trained models (SavedModel format), each in a sub-dir
│   └── mnist_cnn_robust_v1_1.0_20250528_1030_acc0.9910/
│       ├── saved_model.pb
│       ├── variables/
│       ├── assets/
│       ├── training_config_used.yaml                     # Copy of training session config
│       ├── hyperparams_config_used.yaml                  # Copy of hyperparams config
│       ├── training_log.csv                              # Epoch-wise metrics
│       ├── accuracy_plot.png
│       ├── loss_plot.png
│       └── model_card.md                                 # Summary report for this model
├── models_tfjs/                                          # Output: Converted TensorFlow.js models
├── notebooks/                                            # Jupyter notebooks for experimentation & exploration
│   ├── 00_env_check.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_initial_model_training.ipynb
│   └── 03_hyperparam_tuning_yaml.ipynb
├── reports/                                              # Generated reports, figures (beyond individual model logs)
│   └── figures/                                          # Central place for comparison plots, etc.
├── src/
│   ├── __init__.py                                       # Makes src a package
│   ├── cli/                                              # Typer CLI command modules
│   │   ├── __init__.py
│   │   ├── train_cli.py
│   │   ├── predict_cli.py
│   │   ├── model_cli.py
│   │   └── convert_cli.py
│   ├── core/                                             # Core ML logic
│   │   ├── __init__.py
│   │   ├── data_processing.py                            # Data loading, preprocessing, augmentation
│   │   ├── model_architectures.py                        # Model definitions & registry
│   │   ├── training_engine.py                            # Training loops, callbacks, saving
│   │   ├── evaluation_engine.py                          # Evaluation logic
│   │   ├── prediction_engine.py                          # Prediction logic
│   │   └── tfjs_converter_utils.py                       # Utilities for TF.js conversion
│   ├── utils/                                            # Helper utilities
│   │   ├── __init__.py
│   │   ├── config_loader.py                              # Loads YAML configs
│   │   ├── file_ops.py                                   # File/directory operations
│   │   └── logging_setup.py                              # Centralized logging configuration
│   └── main_cli.py                                       # Main Typer app entry point
├── tests/                                                # Unit and integration tests (Future)
│   ├── __init__.py
│   ├── core/
│   └── cli/
├── Makefile                                              # (Optional but Recommended) For automating tasks
├── requirements.txt
└── README.md                                             # This file (Project Documentation part)
```

### **3. Configuration File Design (`configs/`)**

1.  **Hyperparameter Configs (`configs/hyperparams/<name>.yaml`):**
    - Define sets of hyperparameters. Example: `hp_default.yaml`, `hp_robust_cnn_tuned.yaml`.
    - Structure: `learning_rate`, `batch_size`, `optimizer_name` (e.g., 'adam', 'sgd'), architecture-specific params (e.g., `dropout_conv`, `dense_units`).
2.  **Training Session Configs (`configs/training_sessions/<name>.yaml`):**
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
3.  Implement `src/utils/config_loader.py` to load and merge these YAMLs.

### **4. Core Logic Implementation (`src/core/`)**

1.  **`data_processing.py`:**
    - Function `load_mnist_data()`: Returns `(x_train, y_train), (x_test, y_test)`.
    - Function `preprocess_training_data(images, labels, normalize=True)`: Normalizes, reshapes.
    - Function `build_augmentation_pipeline(aug_config)`: Returns a Keras Sequential model of augmentation layers or an `ImageDataGenerator` setup.
    - Function `preprocess_input_image_for_prediction(image_path, target_size=(28,28), invert_colors=False)`: Handles any input image size, resizes with padding, normalizes. Allow `invert_colors` as an option.
2.  **`model_architectures.py`:**
    - Implement `MODEL_REGISTRY`, `register_model` decorator, `get_model_architecture()`, `list_available_model_architectures()`.
    - Define at least two model functions (e.g., `build_simple_cnn`, `build_robust_cnn`).
3.  **`training_engine.py`:**
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
4.  **`evaluation_engine.py`:**
    - `evaluate_model(model_path, x_test, y_test)`: Loads SavedModel, runs `model.evaluate()`, returns metrics.
5.  **`prediction_engine.py`:**
    - `load_trained_model(model_path)`: Loads SavedModel.
    - `predict_digit(model, preprocessed_image_tensor)`: Returns predicted digit and confidence.
6.  **`tfjs_converter_utils.py`:**
    - `convert_saved_model_to_tfjs(saved_model_path, output_tfjs_path)`: Uses `tensorflowjs_converter` CLI via `subprocess`.

### **5. CLI Interface Implementation (`src/cli/` & `src/main_cli.py`)**

1.  **`main_cli.py`:** Setup main Typer app and add sub-apps.
2.  **`train_cli.py`:**
    - `train start --config <path_to_training_session_yaml>`: Calls `training_engine.run_training_session`.
3.  **`model_cli.py`:**
    - `model list-arch`: Calls `model_architectures.list_available_model_architectures`.
    - `model list-trained [--sort-by <accuracy|date>]`: Scans `models_trained/`, parses names, displays info.
    - `model info <trained_model_name_or_path>`: Displays `model_card.md` content, metrics from CSV, path to plots.
    - `model evaluate <trained_model_path>`: Calls `evaluation_engine.evaluate_model`.
4.  **`predict_cli.py`:**
    - `predict image --model <trained_model_path>`: Prompts for image path, calls `data_processing.preprocess_input_image_for_prediction`, then `prediction_engine.predict_digit`.
5.  **`convert_cli.py`:**
    - `convert tfjs --model <trained_model_path> --output <output_dir_for_tfjs_model>`: Calls `tfjs_converter_utils.convert_saved_model_to_tfjs`.
6.  **Utilities (`src/utils/`)**:
    - `logging_setup.py`: Basic logging config.
    - `file_ops.py`: Helpers for scanning directories, ensuring paths exist.

### **6. Robust Reporting Feature (Integrated into `training_engine` and `model_cli`)**

1.  **During Training (`training_engine.py`):**

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

2.  **CLI Display (`model_cli.py model info ...`):**
    - Read and pretty-print the `model_card.md`.
    - Optionally, re-plot graphs if needed or just point to the saved PNGs.

### **7. Makefile (Optional but Recommended for Automation)**

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

### **8. Testing (Future Focus)**

- Plan for unit tests (`pytest`) for core functions in `data_processing.py`, `model_architectures.py`.
- Plan for integration tests for CLI commands (e.g., using `typer.testing.CliRunner`).

---

## **Part 2: Project Documentation (User & Contributor Guide)**

(This section assumes the CLI tool as planned above is fully implemented.)

### **1. Project Overview**

**MNIST Digit Recognizer: CLI Management Tool** is a comprehensive command-line utility designed to streamline the entire lifecycle of developing, training, evaluating, and managing deep learning models for the MNIST handwritten digit recognition task. It provides a structured environment for experimentation, robust model versioning, and easy conversion for deployment.

This tool is built with flexibility in mind, allowing users to define custom model architectures, configure training sessions through simple YAML files, and manage multiple trained models with detailed reporting.

### **2. Key Information**

- **Primary Task:** Handwritten Digit Recognition (0-9)
- **Dataset:** MNIST (Modified National Institute of Standards and Technology database)
- **Core Technology:** Deep Learning (Convolutional Neural Networks)
- **Main Library:** TensorFlow 2.x (with Keras API)
- **CLI Framework:** Typer
- **Configuration:** YAML
- **Key Concepts:** Supervised Learning, Image Classification, CNNs, Hyperparameter Tuning, Model Versioning, Model Deployment (TF.js conversion).

### **3. Features**

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

- Integration with experiment tracking tools (e.g., MLflow, Weights & Biases).
- Support for other image datasets with minor configuration changes.
- Automated hyperparameter optimization (e.g., using KerasTuner).
- More advanced data augmentation techniques.
- Web UI for interacting with the CLI's capabilities (Phase 1 of the larger project).
- Automated testing suite (unit and integration tests).
- Support for distributed training (for larger models/datasets).

### **5. Project Structure**

(Insert the ASCII tree diagram of the `python_mnist_ml/` directory provided in "Project Planning" Section 2 here.)

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

1.  **Input:** The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). Before being fed to the network, these images are preprocessed:

    - **Normalization:** Pixel values (typically 0-255) are scaled to a smaller range (e.g., 0-1). This helps with training stability and convergence.
    - **Reshaping:** Images are reshaped to include a channel dimension (e.g., `(28, 28, 1)` for grayscale).
    - For custom input images during prediction, the tool provides robust preprocessing to convert images of any size to this required format, including grayscale conversion, aspect-ratio preserving resize with padding, and normalization.

2.  **Convolutional Layers (`Conv2D`):**

    - These layers apply a set of learnable filters (kernels) to the input image. Each filter detects specific features like edges, corners, or textures.
    - The output is a set of "feature maps" highlighting these detected features.

3.  **Activation Functions (e.g., `ReLU` - Rectified Linear Unit):**

    - Applied after convolutional layers to introduce non-linearity, allowing the network to learn more complex patterns. ReLU sets all negative values to zero.

4.  **Batch Normalization (`BatchNormalization`):**

    - Normalizes the output of the previous layer during training. This helps stabilize learning, accelerates training, and can act as a regularizer.

5.  **Pooling Layers (`MaxPooling2D`):**

    - Reduce the spatial dimensions (width and height) of the feature maps, making the network more robust to variations in feature positions and reducing computational load. Max pooling takes the maximum value from a small window of the feature map.

6.  **Dropout Layers (`Dropout`):**

    - A regularization technique to prevent overfitting. During training, it randomly sets a fraction of input units to 0 at each update, forcing the network to learn more robust features.

7.  **Flatten Layer (`Flatten`):**

    - Converts the 2D feature maps from the convolutional/pooling blocks into a 1D vector, preparing it for the fully connected layers.

8.  **Dense Layers (Fully Connected Layers):**

    - Standard neural network layers where each neuron is connected to all neurons in the previous layer. These layers learn higher-level combinations of features.
    - The final dense layer has a number of neurons equal to the number of classes (10 for MNIST) and uses a `softmax` activation function.

9.  **Softmax Activation (Output Layer):**

    - Converts the raw output scores (logits) from the final dense layer into a probability distribution over the classes. Each output neuron will have a value between 0 and 1, and all values will sum to 1, representing the model's confidence for each digit.

10. **Training (Optimization):**
    - The model learns by comparing its predictions to the true labels using a **loss function** (e.g., `sparse_categorical_crossentropy` for integer labels).
    - An **optimizer** (e.g., `Adam`, `SGD`) adjusts the model's weights (filters in Conv2D, weights in Dense layers) to minimize this loss function through a process called backpropagation.

### **8. Quickstart Guide**

1.  **Clone the Repository (if applicable) or Setup Project:**
    ```bash
    # Assuming you have the project files in python_mnist_ml/
    cd python_mnist_ml
    ```
2.  **Setup Environment & Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```
3.  **Train a Model with a Default Configuration:**

    - First, ensure a default training session config exists, e.g., `configs/training_sessions/ts_default_run.yaml`.
    - This default config should point to a default hyperparameter config, e.g., `configs/hyperparams/hp_default.yaml`.

    ```bash
    python src/main_cli.py train start --config configs/training_sessions/ts_default_run.yaml
    ```

    - Note the full name of the trained model directory that gets created under `models_trained/`. It will look something like `mnist_CNN_Simple_v1_1.0_YYYYMMDD_HHMM_acc0.XXXX`.

4.  **List Trained Models:**

    ```bash
    python src/main_cli.py model list-trained
    ```

5.  **Predict a Digit from an Image:**
    - You'll need a sample digit image. You can create one or find one online. Save it to `data/02_custom_input_images/my_digit.png`.
    - Replace `<YOUR_TRAINED_MODEL_NAME>` with the directory name from step 3 or `model list-trained`.
    ```bash
    python src/main_cli.py predict image --model models_trained/<YOUR_TRAINED_MODEL_NAME>
    # The CLI will then prompt you for the image path:
    # Enter the path to your image file: data/02_custom_input_images/my_digit.png
    ```

### **9. How-To Guides**

**(This section would contain detailed instructions for each command group and feature)**

- **How to Define a New Model Architecture:**

  1.  Open `src/core/model_architectures.py`.
  2.  Define a new Python function that returns a `tf.keras.Model` (e.g., `build_my_custom_cnn(...)`).
  3.  Decorate it with `@register_model("MyCustomCNN_v1")`.
  4.  You can now reference `"MyCustomCNN_v1"` in your training session YAML files.

- **How to Create New Hyperparameter & Training Session Configs:**

  1.  Navigate to `configs/hyperparams/` or `configs/training_sessions/`.
  2.  Copy an existing `.yaml` file (e.g., `cp hp_default.yaml hp_experimental.yaml`).
  3.  Edit the new YAML file with your desired parameters.
  4.  Reference the new hyperparameter file in your new training session config if needed.
  5.  Use the new training session config with `python src/main_cli.py train start --config ...`.

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

- **Model Architecture Names (in `model_architectures.py` registry):** `Dataset_Purpose_ArchType_Version` (e.g., `MNIST_Classifier_CNN_Simple_v1`, `MNIST_Classifier_CNN_Robust_v1.1`).
- **Hyperparameter Config Files (`configs/hyperparams/`):** `hp_<descriptive_name>.yaml` (e.g., `hp_default.yaml`, `hp_high_lr_sgd.yaml`).
- **Training Session Config Files (`configs/training_sessions/`):** `ts_<descriptive_name>.yaml` (e.g., `ts_robust_cnn_initial_run.yaml`).
- **Trained Model Directories (`models_trained/`):**
  `{dataset_name}_{model_arch_name_from_config}_{model_version_tag_from_config}_{YYYYMMDD}_{HHMM}_acc{val_accuracy:.4f}`
  (e.g., `mnist_CNN_Robust_v1_1.0-alpha_20250528_1430_acc0.9921`).
- **TensorFlow.js Model Directories (`models_tfjs/`):** Typically derived from the trained model name (e.g., `mnist_CNN_Robust_v1_1.0-alpha_20250528_1430_acc0.9921_tfjs`).

### **11. Code Style & Formatting**

- **Python Code:** Formatted using **Black** (default settings).
- **Linting:** Checked with **Flake8**.
- **Docstrings:** Use Google-style Python docstrings.
- **Type Hinting:** Use type hints for function signatures and variables where beneficial for clarity.

### **12. Troubleshooting / FAQ (Example)**

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

### **13. Contributing (Future)**

(Details on how to contribute if this were an open project: code style, testing requirements, pull request process.)

### **14. License**

(Specify a license, e.g., MIT License, Apache 2.0. For personal projects, this is optional but good practice.)

```bash
$ ccds https://github.com/drivendataorg/cookiecutter-data-science
project_name (project_name): My Analysis
repo_name (my_analysis): my_analysis
module_name (my_analysis):
author_name (Your name (or your organization/company/team)): Dat A. Scientist
description (A short description of the project.): This is my analysis of the data.
python_version_number (3.10): 3.12

Select dataset_storage
1 - none
2 - azure
3 - s3
4 - gcs
Choose from [1/2/3/4] (1): 3

bucket (bucket-name): s3://my-aws-bucket

aws_profile (default):

Select environment_manager
1 - virtualenv
2 - conda
3 - pipenv
4 - uv
5 - none
Choose from [1/2/3/4/5] (1): 2

Select dependency_file
1 - requirements.txt
2 - pyproject.toml
3 - environment.yml
4 - Pipfile
Choose from [1/2/3/4] (1): 1

Select pydata_packages
1 - none
2 - basic
Choose from [1/2] (1): 2

Select linting_and_formatting
1 - ruff
2 - flake8+black+isort
Choose from [1/2] (1): 1

Select open_source_license
1 - No license file
2 - MIT
3 - BSD-3-Clause
Choose from [1/2/3] (1): 2

Select docs
1 - mkdocs
2 - none
Choose from [1/2] (1): 1

Select include_code_scaffold
1 - Yes
2 - No
Choose from [1/2] (1): 2
```
