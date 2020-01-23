from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.layers import MaxPooling3D
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers import ZeroPadding3D, Lambda
import keras.backend as K

def contract2d(prev_layer, n_kernel, kernel_size, pool_size, padding,act):
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(prev_layer)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel << 1
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    pool = MaxPooling2D(pool_size=pool_size,strides=pool_size)((conv))
    return conv, pool

def expand2d(prev_layer, left_layer, n_kernel,kernel_size,pool_size, padding,act, dropout=False):
    up = Concatenate(axis=-1)([UpSampling2D(size=pool_size)(prev_layer), left_layer])
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(up)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel >> 1
    if dropout:
        conv = Dropout(.25)(conv)
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    return conv

def contract3d(prev_layer, n_kernel, kernel_size, pool_size, padding,act,pooling):
    conv = ZeroPadding3D(padding=(0, kernel_size//2, kernel_size//2))(prev_layer)
    conv = Conv3D(n_kernel, kernel_size, padding='valid')(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel << 1
    conv = ZeroPadding3D(padding=(0, kernel_size//2, kernel_size//2))(conv)
    conv = Conv3D(n_kernel, kernel_size, padding='valid')(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    pool = MaxPooling3D(pool_size=pool_size,strides=pool_size)((conv))
    return conv, pool

def build_model(input_shape, output_ch=1, dropout=True):
    inputs = Input(input_shape)

    kernel_size = 3
    pool_size = 2
    padding='same'
    activation='relu'
    n_kernel = 16

    pool = inputs
    enc3ds = []
    for _ in range(1):
        conv, pool = contract3d(pool, n_kernel,kernel_size,pool_size,padding,activation,pooling=True)
        enc3ds.append(conv)
        n_kernel = conv.shape[-1].value

    pool = Lambda(lambda x: K.squeeze(x,1))(pool)
    encs = []
    for _ in range(5):
        conv, pool = contract2d(pool, n_kernel,kernel_size,pool_size,padding,activation)
        encs.append(conv)
        n_kernel = conv.shape[-1].value

    for i, enc in enumerate(encs[-2::-1]):
        conv = expand2d(conv, enc, n_kernel,kernel_size,pool_size,padding,activation, dropout=dropout)
        n_kernel = conv.shape[-1].value

    for i, enc in enumerate(enc3ds[-1::-1]):
        enc = Conv3D(n_kernel, (enc.shape[1].value,1,1))(enc) # reduce along z axis
        enc = Lambda(lambda x: K.squeeze(x,1))(enc)
        conv = expand2d(conv, enc, n_kernel,kernel_size,pool_size,padding,activation, dropout=dropout)
        n_kernel = conv.shape[-1].value

    output = Conv2D(output_ch, 1, padding=padding, name='output', activation='softmax')(conv)

    return Model(inputs=inputs, outputs=output)

def visualize_network(model, filename, show_layer_names=False):
    from keras.utils.vis_utils import model_to_dot
    import pydot_ng as pydot
    d = model_to_dot(model, show_shapes=True, show_layer_names=show_layer_names)
    d.set_dpi(50)
    pydot.graph_from_dot_data(d.to_string().replace('None, ','')).write(filename,format='svg')

if __name__ == "__main__":
    model = build_model((6,512,512,1),5)
    visualize_network(model, 'model.svg')
    model.summary()