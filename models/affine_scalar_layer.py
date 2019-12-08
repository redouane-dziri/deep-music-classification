from tensorflow.keras import layers



class AffineScalar(layers.Layer):
	"""This layer multiplies its input by a trainable weight and adds a 
	trainable bias
	"""

	# `suffix` is used for saving purposes to uniquely identify saved weights
	def __init__(self, suffix=0, **kwargs):
		super(AffineScalar, self).__init__(**kwargs)
		self.w = self.add_weight(
			shape=(1,),
			initializer="random_normal",
			trainable=True,
			name=f'affineScalar_w{suffix}'
		)
		self.b = self.add_weight(
			shape=(1,),
			initializer="zeros",
			trainable=True,
			name=f'affineScalar_b{suffix}'
		)
		self.suffix = suffix

	def get_config(self):
		
		config = super().get_config().copy()
		config.update({
            'suffix': self.suffix,
        })
		return config
		
	def call(self, inputs):
		return self.w * inputs + self.b
