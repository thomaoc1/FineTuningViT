import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


class PreprocessedDataset:
    def __init__(self, dataset_name='oxford_iiit_pet:3.*.*'):
        self._dataset, self._metadata = tfds.load(dataset_name, with_info=True, as_supervised=True)
        self._train_dataset, self._test_dataset = self._dataset['train'], self._dataset['test']
        self._preprocess_datasets()

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def _preprocess_datasets(self):
        """
        Preprocess both the training and testing datasets.
        """

        def preprocess_image_vit(image, label):
            image = tf.image.resize(image, [224, 224])
            image: tf.Tensor = tf.cast(image, tf.float32) / 255.0
            mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
            std = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
            image = (image - mean) / std
            return image, label

        self._train_dataset = self._train_dataset.map(preprocess_image_vit)
        self._test_dataset = self._test_dataset.map(preprocess_image_vit)

    def visualize_preprocessed_images(self, num_images=9):
        """
        Visualize preprocessed images from the training dataset.
        """
        plt.figure(figsize=(10, 10))
        for i, (X, y) in enumerate(self._train_dataset.take(num_images)):
            plt.subplot(3, 3, i + 1)
            img = tf.clip_by_value(X, clip_value_min=0.0, clip_value_max=1.0)
            plt.imshow(img)
            plt.title(self._metadata.features['label'].int2str(y).replace('_', ' ').capitalize())
            plt.axis("off")
        plt.show()

