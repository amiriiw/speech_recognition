# Speech Recognition Project

Welcome to the **Speech Recognition Project**! This project is designed to train a model for recognizing voice commands and using it to control a simple Snake game in real-time. The project involves training a TensorFlow model for speech recognition and integrating it with a Pygame-based game.

## Overview

This project consists of two main components:

1. **speech_recognition_model_trainer.py**: This script is responsible for training a TensorFlow model to recognize voice commands. The model is trained using audio data to classify commands such as 'left,' 'right,' 'up,' and 'down.'
2. **speech_recognition_game.py**: This script uses the trained model to control a Snake game based on voice commands. It captures audio from a microphone, processes it to make predictions, and translates those predictions into game controls.

## Libraries Used

The following libraries are used in this project:

- **[tensorflow](https://www.tensorflow.org/)**: TensorFlow is used for building and training the speech recognition model.
- **[numpy](https://numpy.org/)**: NumPy is used for numerical operations and handling arrays.
- **[pygame](https://www.pygame.org/docs/)**: Pygame is used to create the Snake game and handle graphics and user input.
- **[sounddevice](https://python-sounddevice.readthedocs.io/en/0.4.7/)**: Sounddevice is used to capture audio from the microphone in real-time.
- **[queue](https://docs.python.org/3/library/queue.html)**: The Queue module is used for handling audio data in a thread-safe manner.
- **[random](https://docs.python.org/3/library/random.html)**: The Random module is used to generate random positions for the Snake game food.
- **[threading](https://docs.python.org/3/library/threading.html)**: Threading is used to process audio data in parallel with the game loop.

## Detailed Explanation

### `speech_recognition_model_trainer.py`

This script is the core of the project, responsible for training the speech recognition model. The key components of the script are:

- **TrainModel Class**: This class handles the training process for the speech recognition model. It includes methods to load datasets, prepare audio data, build the model, train the model, and evaluate its performance.
  - **_load_commands() Method**: Loads command names from the dataset directory.
  - **load_datasets() Method**: Loads and splits audio datasets into training, validation, and test sets.
  - **_make_spec_ds() Method**: Converts audio data into spectrograms for model input.
  - **build_model() Method**: Defines and compiles the CNN model architecture for speech recognition.
  - **train_model() Method**: Trains the model on the prepared dataset.
  - **evaluate_model() Method**: Evaluates the model's performance on the test set.
- **ExportModel Class**: This class is used to export the trained model for inference. It includes methods to preprocess audio data and make predictions using the trained model.

### `speech_recognition_game.py`

This script integrates the trained model with a real-time game controlled by voice commands. The key components of the script are:

- **SnakeGame Class**: This class sets up and runs the Snake game using Pygame. It handles game initialization, drawing, scoring, and game logic.
- **VoiceControl Class**: This class handles voice command processing and game control. It includes methods for audio processing, making predictions with the model, and translating predictions into game actions.
  - **audio_callback() Method**: Captures audio data from the microphone and adds it to a queue.
  - **process_audio() Method**: Processes audio data, makes predictions using the trained model, and updates the game state based on the predicted commands.

### How It Works

1. **Model Training**:
    - The `speech_recognition_model_trainer.py` script reads audio data from the specified dataset directory.
    - The audio is converted into spectrograms, and the model is trained to classify different voice commands.
    - The trained model is saved for later use.

2. **Voice-Controlled Game**:
    - The `speech_recognition_game.py` script loads the trained model and starts a microphone stream to capture real-time audio.
    - The captured audio is processed and passed through the model to predict voice commands.
    - The predicted commands are used to control the Snake game in real-time.

### Dataset

The dataset used for training the model can be accessed via this [Dataset](https://drive.google.com/drive/folders/1aAST8IX1-3Ri1eBdhq-4oyZVY8gjuuub?usp=sharing).

## Installation and Setup

To use this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/amiriiw/speech_recognition
    cd speech_recognition
    ```

2. Install the required libraries:

    ```bash
    pip install tensorflow numpy pygame sounddevice
    ```

3. Prepare your dataset (an audio dataset with voice commands).

4. Train the model:

    ```bash
    python speech_recognition_model_trainer.py
    ```

5. Run the voice-controlled game:

    ```bash
    python speech_recognition_game.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
