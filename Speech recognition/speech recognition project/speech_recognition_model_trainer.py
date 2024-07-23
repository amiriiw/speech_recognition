"""----------------------------------------------------------------------------------------------------
Welcome, this is amiriiw, this is a simple project about Speech recognition.
This file is the file where we train the model.
----------------------------------------------------------------------------------------------------"""
# Import necessary modules
import pathlib  # https://docs.python.org/3/library/pathlib.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://www.tensorflow.org/
from tensorflow.keras import layers, models  # https://www.tensorflow.org/guide/keras
# -----------------------------------------------------------------------------------


class TrainModel:

    def __init__(self, dataset_path='dataset', seed=42):
        self.dataset_path = dataset_path
        self.seed = seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.data_dir = pathlib.Path(self.dataset_path)
        self.commands = self.load_commands()
        self.label_names = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.model = None

    def load_commands(self):
        commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        return commands[(commands != 'README.md') & (commands != '.DS_Store')]

    def load_datasets(self):
        self.train_ds, self.val_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=self.data_dir,
            batch_size=64,
            validation_split=0.2,
            seed=self.seed,
            output_sequence_length=16000,
            subset='both'
        )
        self.label_names = np.array(self.train_ds.class_names)
        self.train_ds = self.train_ds.map(self.squeeze, tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.map(self.squeeze, tf.data.AUTOTUNE)
        self.test_ds = self.val_ds.shard(num_shards=2, index=0)
        self.val_ds = self.val_ds.shard(num_shards=2, index=1)

    @staticmethod
    def squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    @staticmethod
    def get_spectrogram(waveform):
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def make_spec_ds(self, ds):
        return ds.map(
            map_func=lambda audio, label: (self.get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    def prepare_datasets(self):
        self.train_ds = self.make_spec_ds(self.train_ds).cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
        self.val_ds = self.make_spec_ds(self.val_ds).cache().prefetch(tf.data.AUTOTUNE)
        self.test_ds = self.make_spec_ds(self.test_ds).cache().prefetch(tf.data.AUTOTUNE)

    def build_model(self):
        input_shape = next(iter(self.train_ds.take(1)))[0].shape[1:]
        num_labels = len(self.label_names)

        norm_layer = layers.Normalization()
        norm_layer.adapt(data=self.train_ds.map(map_func=lambda spec, label: spec))

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Resizing(32, 32),
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    def train_model(self, epochs=10):
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )
        return history

    def evaluate_model(self):
        return self.model.evaluate(self.test_ds, return_dict=True)


class ExportModel(tf.Module):

    def __init__(self, model, label_names):
        super().__init__()
        self.model = model
        self.label_names = label_names
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=(), dtype=tf.string))
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        if x.dtype == tf.string:
            x = tf.io.read_file(x)
            x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
            x = tf.squeeze(x, axis=-1)
            x = x[tf.newaxis, :]

        x = TrainModel.get_spectrogram(x)
        result = self.model(x, training=False)

        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(self.label_names, class_ids)
        return {'predictions': result,
                'class_ids': class_ids,
                'class_names': class_names}


train_model = TrainModel()
train_model.load_datasets()
train_model.prepare_datasets()
train_model.build_model()
train_model.train_model(epochs=10)
evaluation_results = train_model.evaluate_model()

export_model = ExportModel(train_model.model, train_model.label_names)
tf.saved_model.save(export_model, "saved_model")
imported = tf.saved_model.load("saved_model")
# -------------------------------------------
