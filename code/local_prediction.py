# This file is used to do local predictions, because the tf.function decorator
# needs the tensorflow imported. Importing tensorflow reserves VRAM, which is
# only needed for local predictions. That's why it's in a separate file.

import tensorflow as tf

@tf.function
def predict_local(model, args):
	return model(args)