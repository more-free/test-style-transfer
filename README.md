# test-style-transfer

I stricly followed [the CVPR paper by Gatys, Ecker and Bethge](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), implemented the algorithm using tensorflow and VGG19, with the following parameter settings : 

```
content_weight = 0.0001,  
style_weight = 0.1 ~ 1.0
content_layer = 'relu4_2'
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
```

Here are some results I got, 

content image : <b>tubingen hotels</b>

![tubingen hotels](/tubingen-small.jpg)


style image : <b>the starry night</b>

![starry night](/1-style.jpg)

and this is visually the best result I got(`content_weight = 0.0001, style_weight = 0.5, optimizer = Adam optimizer, start_learning_rate = 10, iteration = 250`)

![starry-night-tubingen](/style_transfer_iter_250.jpg)

However I found in the [original paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf),  or in [Justin's implementation](https://github.com/jcjohnson/neural-style), visually the result is much more appealing. 

result borrowed from Justin's repo :

![justin-result](https://github.com/jcjohnson/neural-style/blob/master/examples/outputs/tubingen_starry.png)

and this is the result from the original paper : 
![cvpr-result](/starry-night-cvpr.png)

the painting style of starry-night is amazaingly blended ! how did they achieve this ?


also, when I tried to reconstruct the style from `style loss function` defined in the paper, I got less-appealing results either (this is already the best result I got in my opionion):

![starry-style](/style_reconstruction_380.jpg)

and this is the original style reconstruction from the CVPR paper which shows much better moon

![starry-style-cvpr](/starry-night-style-cvpr.png)

i am still cleaning my code, but here is the code for computing style loss : 

```python
import tensorflow as tf

def style_loss(style_layer, style_cnn, transferred_cnn):
    G, shape = gram_matrix(style_cnn, style_layer)
    A, _ = gram_matrix(transferred_cnn, style_layer)
    N, M = int(shape[1]), int(shape[0]) # N = number of filters,  M = size of activation of filters (feature map)

    return tf.nn.l2_loss((G - A) / (N * M)) / 2.0

def gram_matrix(cnn, style_layer):
    _, _, _, channels = cnn[style_layer].get_shape()
    feature_map = tf.reshape(cnn[style_layer], [-1, int(channels)])
    return tf.matmul(tf.transpose(feature_map), feature_map), feature_map.get_shape()

def weighted_style_loss(style_layer_weights, style_layers, style_cnn, transferred_cnn):
    layered_style_loss = [style_loss(layer, style_cnn, transferred_cnn) for layer in style_layers]
    return sum([w * s for (w, s) in zip(style_layer_weights, layered_style_loss)])
```
