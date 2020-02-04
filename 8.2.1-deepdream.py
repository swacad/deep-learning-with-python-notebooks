from keras.applications import inception_v3
from keras import backend as K
import numpy as np

import scipy
from keras.preprocessing import image

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

def preprocess_image(image_path):
    """ Utility function to open, reisize, and format pics into tensors that inception v3 can process
    """
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    """Util function to convert tensor into valid image"""
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:  # Undo preprocessing that was performed by inception_v3.preprocess_input
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

K.set_learning_phase(0)  # Disable all training specific operations

# Build Inception V3 network without convolutional base and load with pretrained ImageNet weights.
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# Dict mapping layer names to a coefficient quantifying how much the layer's activation contributes to the loss
# you want to maximize. Layer names are hard coded and layer names can be listed with model.summary()
layer_contributions = {'mixed2': 0.2,
                       'mixed3': 3.0,
                       'mixed4': 2.0,
                       'mixed5': 1.5}

layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(layer_dict)

loss = K.variable(0.0)

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling


dream = model.input  # dream tensor holds the generated image

grads = K.gradients(loss, dream)[0]  # computes the gradients with regard to loss

grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # normalizes the gradients

outputs = [loss, grads]  # Setup Keras function to retrieve the value of the loss and gradients.
fetch_loss_and_grads = K.function([dream], outputs) 

# Run the gradient ascents
def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    
    return x

import numpy as np

# Change for different effects
step = 0.01  # gradient ascent step size
num_octave = 3  #  Number of scales to run 
octave_scale = 1.4  # size ratio between scales
iterations = 20  # Number of ascent steps at each scale

max_loss = 10. # Stop if loss grows larger than max_loss

base_image_path = 'star.gif'  # path of image

img = preprocess_image(base_image_path)  # load into np array

original_shape = img.shape[1:3]

# prepare list of shape tuples defining the different scales to run gradient ascent
successive_shapes = [original_shape]                                      
for i in range(1, num_octave):                                            
    shape = tuple([int(dim / (octave_scale ** i))                         
    for dim in original_shape])                                          
    successive_shapes.append(shape)                                       

successive_shapes = successive_shapes[::-1]  # Reverse list of shapes in increasing order

original_img = np.copy(img)  # resize array to smallest scale
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)  # scale up dream image

    # run gradient ascent, altering the dream
    img = gradient_ascent(img,
                          iterations=iterations,  
                          step=step,                                      
                          max_loss=max_loss)                           

    # scale up smaller version of original image it will be pixellated.
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)  # compute HQ version of original image at that size

    # The difference between the two is the detail that was lost when scaling up.
    lost_detail = same_size_original - upscaled_shrunk_original_img  

    img += lost_detail  # Reinject lost detail into the dream.
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

save_img(img, fname='final_dream.png')

