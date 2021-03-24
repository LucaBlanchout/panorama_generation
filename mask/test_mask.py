import matplotlib.pyplot as plt
from roi import RoiPoly
from skylibs.envmap import EnvironmentMap


def create_mask(img):
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.title("left click: line segment         right click: close region")
    plt.show(block=False)

    roi1 = RoiPoly(color='r', fig=fig)

    mask = (roi1.get_mask(img[:, :, 0])).astype(float)

    return mask


def create_latlong_mask(index):
    width_resolution = 1024
    number_of_cameras = 3

    in_path = 'images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/real/360render_'

    img = plt.imread(in_path + str(index) + '.jpg')

    mask = create_mask(img)

    plt.imshow(mask)
    plt.show()

    fname = 'images/1024/3/real/360render_' + str(index) + '_mask_latlong.jpg'
    plt.imsave(fname, mask)


def convert_mask_to_cubemap(index):
    width_resolution = 1024
    number_of_cameras = 3
    envmap_type = 'cube'

    in_path = 'images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/real/360render_'
    envmap = EnvironmentMap(in_path + str(index) + '_mask_latlong.jpg', 'latlong')
    envmap = envmap.convertTo(envmap_type)

    fname = 'images/1024/3/real/360render_' + str(index) + '_mask_cubemap.jpg'
    plt.imsave(fname, envmap.data)


if __name__ == "__main__":
    index = 2
    create_latlong_mask(index)
    convert_mask_to_cubemap(index)



