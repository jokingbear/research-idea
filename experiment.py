from keras import layers, models
from custom_modules import con_block, res_block
from keras.utils import plot_model

x = layers.Input((28, 28, 1))

con0 = x
con0 = con_block(con0, 32, 3, normalization=layers.BatchNormalization())

con1 = res_block(con0, down_sample=True)
con1 = res_block(con1)

con2 = res_block(con1, down_sample=True)
con2 = res_block(con2)

flat = layers.GlobalAvgPool2D()(con2)

model = models.Model(x, flat)

plot_model(model, show_shapes=True)



