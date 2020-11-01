from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.keras.utils import conv_utils
from etmiss_utils import get_phi_tensor

phi_x_tf, phi_y_tf = get_phi_tensor()

def add_wrap(x, half_kernel_size=1, const=0., data_format='channels_last'):

    '''
    applies const padding in eta ('hard wall boundary conditions), and associates pi with -pi in phi
    (periodic boundary conditions). Padding is necessary to preserve the dimension of the image as it
    passes through the network.
    '''
    if data_format == 'channels_last':
        x = K.permute_dimensions(x, (0,3,1,2))

    wrap_pad = K.concatenate([x[:, :, :, -half_kernel_size:],
                             x,
                             x[:, :, :, :half_kernel_size]],
                             axis=3)
    const_pad = K.ones_like(wrap_pad[:, :, :half_kernel_size, :]) * const
    padded = K.concatenate([const_pad,
                         wrap_pad,
                         const_pad],
                         axis=2)
                         
    if data_format == 'channels_last':
        return K.permute_dimensions(padded, (0,2,3,1))
    else:
        return padded
                         
def Wrap(half_kernel_size=1,const=0.):
    return layers.Lambda(lambda x: add_wrap(x,half_kernel_size=half_kernel_size,const=const))
    
class Conv2D_fb(layers.Conv2D):
    '''
    Convolutional layer with a weights matrix which is symmetric in eta
    '''

    def call(self, inputs):
        print(self.kernel)
        outputs_plus = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        
        outputs_minus = K.conv2d(
                inputs,
                K.reverse(self.kernel,0),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs_plus = K.bias_add(
                outputs_plus,
                self.bias,
                data_format=self.data_format)
            
            outputs_minus = K.bias_add(
                outputs_minus,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return [self.activation(outputs_plus),self.activation(outputs_minus)]
        
        return [outputs_plus,outputs_minus]
    
    def compute_output_shape(self,input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
            
        if self.data_format == 'channels_last':
            return [(input_shape[0],) + tuple(new_space) + (self.filters,),
                    (input_shape[0],) + tuple(new_space) + (self.filters,)]
        elif self.data_format == 'channels_first':
            return [(input_shape[0], self.filters) + tuple(new_space), 
                    (input_shape[0], self.filters) + tuple(new_space)]