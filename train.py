import config
import tensorflow as tf
from keras.models import Model

class Trainer:
	def __init__(self, model: Model):
		self.model = model

		self.batch_size = config.BATCH_SIZE


	def sample_batch(self):
		pass

	def train_model(self):
		batch = self.sample_batch()
		self.model.train_on_batch(x=batch[0], y=batch[1], return_dict=True)
