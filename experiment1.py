from implementations import blocks

from keras import layers, Model, utils

x = layers.Input([256, 256, 1])

con0 = blocks.con_block("con0", 64, normalization=layers.BatchNormalization())(x)

con1 = blocks.res_block("down_res1", 32, 64, 4, down_sample=True, normalizations=[layers.BatchNormalization(),
                                                                                  layers.BatchNormalization(),
                                                                                  layers.BatchNormalization()])(con0)
con1 = blocks.con_block("con1", 128, normalization=layers.BatchNormalization())(con1)
con1 = blocks.res_block("res1", 32, 128, 4, normalizations=[layers.BatchNormalization(),
                                                            layers.BatchNormalization(),
                                                            layers.BatchNormalization()])(con1)

model = Model(x, con1)

utils.plot_model(model, show_shapes=True)
