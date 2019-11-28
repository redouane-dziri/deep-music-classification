from tensorflow.keras import layers



class AffineScalar(layers.Layer):
	"""[summary]
	
	Arguments:
		layers {[type]} -- [description]
	"""

	def __init__(self, **kwargs):
		super(AffineScalar, self).__init__(**kwargs)
		self.w = self.add_weight(
			shape=(1,),
			initializer="random_normal",
			trainable=True,
			name='affineScalar_w'
		)
		self.b = self.add_weight(
			shape=(1,),
			initializer="zeros",
			trainable=True,
			name='affineScalar_b'
		)
		
	def call(self, inputs):
		return self.w * inputs + self.b
