import imageio


# copied from https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/utils.py
def generate_animation(path, num) :
    images = []
    for e in range(num) :
        img_name = path +'_epoch%03d' %(e+1) + '.jpg'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_animation.gif', images, fps = 5)
    
    
def tensor2image(tensor) :
    return tensor.permute(1,2,0).detach().cpu().numpy()