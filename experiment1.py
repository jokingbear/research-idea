from implementations import models
from keras.utils import plot_model

model = models.routing_encoder(f=64)

plot_model(model, show_shapes=True)
