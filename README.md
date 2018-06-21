# neural-style-transfer
artistic style transfer using cnn 

# basic
1. the code uses a pre trained VGG19 architecture for performing neural style transfer that can be downloaded from http://www.vlfeat.org/matconvnet/pretrained/
2. the inspiration was taken from the paper A Neural Algorithm of Artistic Style (https://arxiv.org/pdf/1508.06576.pdf)
3. some tweaking of parameters in loss functions were done to gentrate a fairly good results.
4. the code is a tensorflow implementation that can run on a cuda enabled GPU (my specs : 2GB Nvidia 940MX)
5. in the VGG architecture the maxpooling is repaced by average pooling as it proved to be better according to above mentioned paper.

# some results :

![alt text](https://github.com/jaydeepthik/neural-style-transfer/blob/master/output/generated_image.jpg)
