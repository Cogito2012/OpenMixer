#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import cv2
import torch


def clip_boxes_to_image(boxes, height, width):
    """
    Clip the boxes with the height and width of the image size.
    Args:
        boxes (ndarray): bounding boxes to peform crop. The dimension is
        `num boxes` x 4.
        height (int): the height of the image.
        width (int): the width of the image.
    Returns:
        boxes (ndarray): cropped bounding boxes.
    """
    boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return boxes


def clip_boxes_tensor(boxes, height, width):
    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0., max=width-1)
    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0., max=height-1)
    return boxes


def random_short_side_scale_jitter_list(images, min_size, max_size, boxes=None):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (list): list of images to perform scale jitter. Dimension is
            `height` x `width` x `channel`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        boxes (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    Returns:
        (list): the list of scaled images with dimension of
            `new height` x `new width` x `channel`.
        (ndarray or None): the scaled boxes with dimension of
            `num boxes` x 4.
    """
    size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))

    height = images[0].shape[0]
    width = images[0].shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = [
                proposal * float(new_height) / height for proposal in boxes
            ]
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = [proposal * float(new_width) / width for proposal in boxes]
    return (
        [
            cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            for image in images
        ],
        boxes,
    )


def random_short_side_scale_jitter(images, min_sizes, max_size, boxes=None):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (list): list of images to perform scale jitter. Dimension is
            `height` x `width` x `channel`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        boxes (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    Returns:
        (list): the list of scaled images with dimension of
            `new height` x `new width` x `channel`.
        (ndarray or None): the scaled boxes with dimension of
            `num boxes` x 4.
    """

    min_size = np.random.choice(min_sizes)

    height = images[0].shape[0]
    width = images[0].shape[1]
    if (width <= height <=max_size and width == min_size) or (
        height <= width <= max_size and height == min_size
    ):
        return images, boxes

    new_width = min_size
    new_height = min_size
    if width < height:
        new_height = int((float(height) / width) * min_size + 0.5)
        if boxes is not None:
            boxes = [proposal * float(new_width) / width for proposal in boxes]
    else:
        new_width = int((float(width) / height) * min_size + 0.5)
        if boxes is not None:
            boxes = [proposal * float(new_height) / height for proposal in boxes]
    return (
        [
            cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            for image in images
        ],
        boxes,
    )


def get_padding_margins(input_size, target_size, padding='center'):
    height, width = input_size
    if width < height:
        mt, mb = 0, 0
        if padding == 'center':
            ml = int((target_size[1] - width) // 2)
            mr = target_size[1] - ml - width
        else:  #  padding == 'left'
            ml, mr = 0, target_size[1] - width
    else:
        ml, mr = 0, 0
        if padding == 'center':
            mt = int((target_size[0] - height) // 2)
            mb = target_size[0] - mt - height
        else:  # padding == 'top'
            mt, mb = 0, target_size[0] - height
    return mt, mb, ml, mr


def long_side_scale_and_pad(images, target_size=[224, 224], boxes=None, padding='center'):
    """ First scaling the images to have the longer side (height, or width) the same as target height (or width),
        Then, padding the image with its mean pixel
    """
    height = images[0].shape[0]
    width = images[0].shape[1]
    
    new_height, new_width = target_size[0], target_size[1]
    if width < height:
        new_width = int((float(width) / height) * new_height + 0.5)
        if boxes is not None:
            boxes = [proposal * float(new_height) / height for proposal in boxes]
    else:
        new_height = int((float(height) / width) * new_width + 0.5)
        if boxes is not None:
            boxes = [proposal * float(new_width) / width for proposal in boxes]
    new_images = [
        cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) 
        for image in images]
    
    # padding margins
    mt, mb, ml, mr = get_padding_margins([new_height, new_width], target_size, padding=padding)
    new_images = [
        cv2.copyMakeBorder(image, mt, mb, ml, mr, cv2.BORDER_CONSTANT, value=0)
        for image in new_images
    ]
    if boxes is not None: # for center padding, box coordinates need to be updated (ml or mt is non-zero)
        boxes = [proposal + np.array([ml, mt, ml, mt]) for proposal in boxes]
    
    return new_images, boxes


def scale(size, image):
    """
    Scale the short side of the image to size.
    Args:
        size (int): size to scale the image.
        image (array): image to perform short side scale. Dimension is
            `height` x `width` x `channel`.
    Returns:
        (ndarray): the scaled image with dimension of
            `height` x `width` x `channel`.
    """
    height = image.shape[0]
    width = image.shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return img.astype(np.float32)


def scale_boxes(size, boxes, height, width):
    """
    Scale the short side of the box to size.
    Args:
        size (int): size to scale the image.
        boxes (ndarray): bounding boxes to peform scale. The dimension is
        `num boxes` x 4.
        height (int): the height of the image.
        width (int): the width of the image.
    Returns:
        boxes (ndarray): scaled bounding boxes.
    """
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return boxes

    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        boxes *= float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        boxes *= float(new_width) / width
    return boxes


def reverse_scale_boxes(size, boxes, height, width):
    """
    height, width is torch.Tensor with dtype, torch.int64.
    For pytorch1.5 (and maybe below 1.5), some bad performance like following:
    e.g:   1.0 / Tensor([600], dtype=int64) -->  0

    For pytorch1.7, this is resolved. So we add float(height) and float(width)
    for number stable
    """
    height = float(height)
    width = float(width)
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        boxes /= float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        boxes /= float(new_width) / width
    return boxes


def horizontal_flip_list(prob, images, order="CHW", boxes=None):
    """
    Horizontally flip the list of image and optional boxes.
    Args:
        prob (float): probability to flip.
        image (list): ilist of images to perform short side scale. Dimension is
            `height` x `width` x `channel` or `channel` x `height` x `width`.
        order (str): order of the `height`, `channel` and `width`.
        boxes (list): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        (ndarray): the scaled image with dimension of
            `height` x `width` x `channel`.
        (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    """
    _, width, _ = images[0].shape
    if np.random.uniform() < prob:
        if boxes is not None:
            boxes = [flip_boxes(proposal, width) for proposal in boxes]
        if order == "CHW":
            out_images = []
            for image in images:
                image = np.asarray(image).swapaxes(2, 0)
                image = image[::-1]
                out_images.append(image.swapaxes(0, 2))
            return out_images, boxes
        elif order == "HWC":
            return [cv2.flip(image, 1) for image in images], boxes
    return images, boxes


def spatial_shift_crop_list(size, images, spatial_shift_pos, boxes=None):
    """
    Perform left, center, or right crop of the given list of images.
    Args:
        size (int): size to crop.
        image (list): ilist of images to perform short side scale. Dimension is
            `height` x `width` x `channel` or `channel` x `height` x `width`.
        spatial_shift_pos (int): option includes 0 (left), 1 (middle), and
            2 (right) crop.
        boxes (list): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (ndarray): the cropped list of images with dimension of
            `height` x `width` x `channel`.
        boxes (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    """

    assert spatial_shift_pos in [0, 1, 2]

    height = images[0].shape[0]
    width = images[0].shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_shift_pos == 0:
            y_offset = 0
        elif spatial_shift_pos == 2:
            y_offset = height - size
    else:
        if spatial_shift_pos == 0:
            x_offset = 0
        elif spatial_shift_pos == 2:
            x_offset = width - size

    cropped = [
        image[y_offset : y_offset + size, x_offset : x_offset + size, :]
        for image in images
    ]
    assert cropped[0].shape[0] == size, "Image height not cropped properly"
    assert cropped[0].shape[1] == size, "Image width not cropped properly"

    if boxes is not None:
        for i in range(len(boxes)):
            boxes[i][:, [0, 2]] -= x_offset
            boxes[i][:, [1, 3]] -= y_offset
    return cropped, boxes


def reverse_spatial_shift_crop_list(size, ori_height, ori_width, spatial_shift_pos, boxes=None):
    assert spatial_shift_pos == 1, "Not Implement for 0, 2"

    height = float(ori_height)
    width = float(ori_width)
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        new_height, new_width = height, width
    else:
        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

    y_offset = int(math.ceil((new_height - size) / 2))
    x_offset = int(math.ceil((new_width - size) / 2))

    boxes[:, [0, 2]] += x_offset
    boxes[:, [1, 3]] += y_offset

    return boxes


def CHW2HWC(image):
    """
    Transpose the dimension from `channel` x `height` x `width` to
        `height` x `width` x `channel`.
    Args:
        image (array): image to transpose.
    Returns
        (array): transposed image.
    """
    return image.transpose([1, 2, 0])


def HWC2CHW(image):
    """
    Transpose the dimension from `height` x `width` x `channel` to
        `channel` x `height` x `width`.
    Args:
        image (array): image to transpose.
    Returns
        (array): transposed image.
    """
    return image.transpose([2, 0, 1])


def color_jitter_list(
    images, img_brightness=0, img_contrast=0, img_saturation=0
):
    """
    Perform color jitter on the list of images.
    Args:
        images (list): list of images to perform color jitter.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (list): the jittered list of images.
    """
    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_list(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_list(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_list(img_saturation, images)
    return images


def lighting_list(imgs, alphastd, eigval, eigvec, alpha=None):
    """
    Perform AlexNet-style PCA jitter on the given list of images.
    Args:
        images (list): list of images to perform lighting jitter.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (list): eigenvalues for PCA jitter.
        eigvec (list[list]): eigenvectors for PCA jitter.
    Returns:
        out_images (list): the list of jittered images.
    """
    if alphastd == 0:
        return imgs
    # generate alpha1, alpha2, alpha3
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = []
    for img in imgs:
        for idx in range(img.shape[0]):
            img[idx] = img[idx] + rgb[2 - idx]
        out_images.append(img)
    return out_images


def color_normalization(image, mean, stddev):
    """
    Perform color normalization on the image with the given mean and stddev.
    Args:
        image (array): image to perform color normalization.
        mean (float): mean value to subtract.
        stddev (float): stddev to devide.
    """
    # Input image should in format of CHW
    assert len(mean) == image.shape[0], "channel mean not computed properly"
    assert len(stddev) == image.shape[0], "channel stddev not computed properly"
    for idx in range(image.shape[0]):
        image[idx] = image[idx] - mean[idx]
        image[idx] = image[idx] / stddev[idx]
    return image


def pad_image(image, pad_size, order="CHW"):
    """
    Pad the given image with the size of pad_size.
    Args:
        image (array): image to pad.
        pad_size (int): size to pad.
        order (str): order of the `height`, `channel` and `width`.
    Returns:
        img (array): padded image.
    """
    if order == "CHW":
        img = np.pad(
            image,
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode=str("constant"),
        )
    elif order == "HWC":
        img = np.pad(
            image,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode=str("constant"),
        )
    return img


def horizontal_flip(prob, image, order="CHW"):
    """
    Horizontally flip the image.
    Args:
        prob (float): probability to flip.
        image (array): image to pad.
        order (str): order of the `height`, `channel` and `width`.
    Returns:
        img (array): flipped image.
    """
    assert order in ["CHW", "HWC"], "order {} is not supported".format(order)
    if np.random.uniform() < prob:
        if order == "CHW":
            image = image[:, :, ::-1]
        elif order == "HWC":
            image = image[:, ::-1, :]
        else:
            raise NotImplementedError("Unknown order {}".format(order))
    return image


def flip_boxes(boxes, im_width):
    """
    Horizontally flip the boxes.
    Args:
        boxes (array): box to flip.
        im_width (int): width of the image.
    Returns:
        boxes_flipped (array): flipped box.
    """

    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped


def crop_boxes(boxes, x_offset, y_offset):
    """
    Crop the boxes given the offsets.
    Args:
        boxes (array): boxes to crop.
        x_offset (int): offset on x.
        y_offset (int): offset on y.
    """
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset
    return boxes


def random_crop_list(images, size, pad_size=0, order="CHW", boxes=None):
    """
    Perform random crop on a list of images.
    Args:
        images (list): list of images to perform random crop.
        size (int): size to crop.
        pad_size (int): padding size.
        order (str): order of the `height`, `channel` and `width`.
        boxes (list): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (ndarray): the cropped list of images with dimension of
            `height` x `width` x `channel`.
        boxes (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    """
    # explicitly dealing processing per image order to avoid flipping images.
    if pad_size > 0:
        images = [
            pad_image(pad_size=pad_size, image=image, order=order)
            for image in images
        ]

    # image format should be CHW.
    if order == "CHW":
        if images[0].shape[1] == size and images[0].shape[2] == size:
            return images, boxes
        height = images[0].shape[1]
        width = images[0].shape[2]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = [
            image[:, y_offset : y_offset + size, x_offset : x_offset + size]
            for image in images
        ]
        assert cropped[0].shape[1] == size, "Image not cropped properly"
        assert cropped[0].shape[2] == size, "Image not cropped properly"
    elif order == "HWC":
        if images[0].shape[0] == size and images[0].shape[1] == size:
            return images, boxes
        height = images[0].shape[0]
        width = images[0].shape[1]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = [
            image[y_offset : y_offset + size, x_offset : x_offset + size, :]
            for image in images
        ]
        assert cropped[0].shape[0] == size, "Image not cropped properly"
        assert cropped[0].shape[1] == size, "Image not cropped properly"

    if boxes is not None:
        boxes = [crop_boxes(proposal, x_offset, y_offset) for proposal in boxes]
    return cropped, boxes


def center_crop(size, image):
    """
    Perform center crop on input images.
    Args:
        size (int): size of the cropped height and width.
        image (array): the image to perform center crop.
    """
    height = image.shape[0]
    width = image.shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    cropped = image[y_offset : y_offset + size, x_offset : x_offset + size, :]
    assert cropped.shape[0] == size, "Image height not cropped properly"
    assert cropped.shape[1] == size, "Image width not cropped properly"
    return cropped


# ResNet style scale jittering: randomly select the scale from
# [1/max_size, 1/min_size]
def random_scale_jitter(image, min_size, max_size):
    """
    Perform ResNet style random scale jittering: randomly select the scale from
        [1/max_size, 1/min_size].
    Args:
        image (array): image to perform random scale.
        min_size (int): min size to scale.
        max_size (int) max size to scale.
    Returns:
        image (array): scaled image.
    """
    img_scale = int(
        round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
    )
    image = scale(img_scale, image)
    return image


def random_scale_jitter_list(images, min_size, max_size):
    """
    Perform ResNet style random scale jittering on a list of image: randomly
        select the scale from [1/max_size, 1/min_size]. Note that all the image
        will share the same scale.
    Args:
        images (list): list of images to perform random scale.
        min_size (int): min size to scale.
        max_size (int) max size to scale.
    Returns:
        images (list): list of scaled image.
    """
    img_scale = int(
        round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
    )
    return [scale(img_scale, image) for image in images]


def random_sized_crop(image, size, area_frac=0.08):
    """
    Perform random sized cropping on the given image. Random crop with size
        8% - 100% image area and aspect ratio in [3/4, 4/3].
    Args:
        image (array): image to crop.
        size (int): size to crop.
        area_frac (float): area of fraction.
    Returns:
        (array): cropped image.
    """
    for _ in range(0, 10):
        height = image.shape[0]
        width = image.shape[1]
        area = height * width
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w, h = h, w
        if h <= height and w <= width:
            if height == h:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, height - h)
            if width == w:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, width - w)
            y_offset = int(y_offset)
            x_offset = int(x_offset)
            cropped = image[y_offset : y_offset + h, x_offset : x_offset + w, :]
            assert (
                cropped.shape[0] == h and cropped.shape[1] == w
            ), "Wrong crop size"
            cropped = cv2.resize(
                cropped, (size, size), interpolation=cv2.INTER_LINEAR
            )
            return cropped.astype(np.float32)
    return center_crop(size, scale(size, image))


def lighting(img, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on the given image.
    Args:
        image (array): list of images to perform lighting jitter.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (array): eigenvalues for PCA jitter.
        eigvec (list): eigenvectors for PCA jitter.
    Returns:
        img (tensor): the jittered image.
    """
    if alphastd == 0:
        return img
    # generate alpha1, alpha2, alpha3.
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    for idx in range(img.shape[0]):
        img[idx] = img[idx] + rgb[2 - idx]
    return img


def random_sized_crop_list(images, size, crop_area_fraction=0.08):
    """
    Perform random sized cropping on the given list of images. Random crop with
        size 8% - 100% image area and aspect ratio in [3/4, 4/3].
    Args:
        images (list): image to crop.
        size (int): size to crop.
        area_frac (float): area of fraction.
    Returns:
        (list): list of cropped image.
    """
    for _ in range(0, 10):
        height = images[0].shape[0]
        width = images[0].shape[1]
        area = height * width
        target_area = np.random.uniform(crop_area_fraction, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w, h = h, w
        if h <= height and w <= width:
            if height == h:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, height - h)
            if width == w:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, width - w)
            y_offset = int(y_offset)
            x_offset = int(x_offset)

            croppsed_images = []
            for image in images:
                cropped = image[
                    y_offset : y_offset + h, x_offset : x_offset + w, :
                ]
                assert (
                    cropped.shape[0] == h and cropped.shape[1] == w
                ), "Wrong crop size"
                cropped = cv2.resize(
                    cropped, (size, size), interpolation=cv2.INTER_LINEAR
                )
                croppsed_images.append(cropped.astype(np.float32))
            return croppsed_images

    return [center_crop(size, scale(size, image)) for image in images]


def blend(image1, image2, alpha):
    return image1 * alpha + image2 * (1 - alpha)


def grayscale(image):
    """
    Convert the image to gray scale.
    Args:
        image (tensor): image to convert to gray scale. Dimension is
            `channel` x `height` x `width`.
    Returns:
        img_gray (tensor): image in gray scale.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = np.copy(image)
    gray_channel = 0.299 * image[2] + 0.587 * image[1] + 0.114 * image[0]
    img_gray[0] = gray_channel
    img_gray[1] = gray_channel
    img_gray[2] = gray_channel
    return img_gray


def saturation(var, image):
    """
    Perform color saturation on the given image.
    Args:
        var (float): variance.
        image (array): image to perform color saturation.
    Returns:
        (array): image that performed color saturation.
    """
    img_gray = grayscale(image)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def brightness(var, image):
    """
    Perform color brightness on the given image.
    Args:
        var (float): variance.
        image (array): image to perform color brightness.
    Returns:
        (array): image that performed color brightness.
    """
    img_bright = np.zeros(image.shape).astype(image.dtype)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_bright, alpha)


def contrast(var, image):
    """
    Perform color contrast on the given image.
    Args:
        var (float): variance.
        image (array): image to perform color contrast.
    Returns:
        (array): image that performed color contrast.
    """
    img_gray = grayscale(image)
    img_gray.fill(np.mean(img_gray[0]))
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def saturation_list(var, images):
    """
    Perform color saturation on the list of given images.
    Args:
        var (float): variance.
        images (list): list of images to perform color saturation.
    Returns:
        (list): list of images that performed color saturation.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def brightness_list(var, images):
    """
    Perform color brightness on the given list of images.
    Args:
        var (float): variance.
        images (list): list of images to perform color brightness.
    Returns:
        (array): list of images that performed color brightness.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_bright = np.zeros(image.shape).astype(image.dtype)
        out_images.append(blend(image, img_bright, alpha))
    return out_images


def contrast_list(var, images):
    """
    Perform color contrast on the given list of images.
    Args:
        var (float): variance.
        images (list): list of images to perform color contrast.
    Returns:
        (array): image that performed color contrast.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        img_gray.fill(np.mean(img_gray[0]))
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def color_jitter(image, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perform color jitter on the given image.
    Args:
        image (array): image to perform color jitter.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        image (array): the jittered image.
    """
    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                image = brightness(img_brightness, image)
            elif jitter[order[idx]] == "contrast":
                image = contrast(img_contrast, image)
            elif jitter[order[idx]] == "saturation":
                image = saturation(img_saturation, image)
    return image


def revert_scaled_boxes(size, boxes, img_height, img_width):
    """
    Revert scaled input boxes to match the original image size.
    Args:
        size (int): size of the cropped image.
        boxes (array): shape (num_boxes, 4).
        img_height (int): height of original image.
        img_width (int): width of original image.
    Returns:
        reverted_boxes (array): boxes scaled back to the original image size.
    """
    scaled_aspect = np.min([img_height, img_width])
    scale_ratio = scaled_aspect / size
    reverted_boxes = boxes * scale_ratio
    return reverted_boxes


def postprocessing_box_tensor(boxes, is_eval, norm_scale, crop_size, ori_height, ori_width):
    if is_eval:
        boxes = reverse_spatial_shift_crop_list(
            size=crop_size,
            ori_height=ori_height,
            ori_width=ori_width,
            spatial_shift_pos=1,
            boxes=boxes,
        )
    boxes = reverse_scale_boxes(crop_size, boxes, ori_height, ori_width)

    boxes = clip_boxes_tensor(boxes, ori_height, ori_width)
    if norm_scale:
        ori_whwh = torch.tensor([ori_width, ori_height, ori_width, ori_height],
                                dtype=torch.float32, device=boxes.device)
        boxes = boxes / ori_whwh
    return boxes


def detector_postprocess(boxes, ori_h, ori_w, cur_h, cur_w, norm_scale):
    """
    replace the fun: postprocessing_box_tensor
    used to post process predicted box when using the detectron2
    data transform backend
    :height  original image height
    :width original image width
    """
    ori_h = float(ori_h)  # for numerically stable
    ori_w = float(ori_w)

    scale_x = ori_w / cur_w
    scale_y = ori_h / cur_h
    # scale box to original image size
    boxes[:, 0::2] *= scale_x
    boxes[:, 1::2] *= scale_y

    # clip box
    assert torch.isfinite(boxes).all(), "Box tensor contains infinite or NaN"
    boxes[:, 0].clamp_(min=0, max=ori_w)
    boxes[:, 1].clamp_(min=0, max=ori_h)
    boxes[:, 2].clamp_(min=0, max=ori_w)
    boxes[:, 3].clamp_(min=0, max=ori_h)

    if norm_scale:
        # AVA evaluator need the normalized box
        boxes[:, 0::2] /= ori_w
        boxes[:, 1::2] /= ori_h

    return boxes



class PreprocessWithBoxes:
    def __init__(self, split, cfg_data, cfg_dataset):
        self._split = split
        
        self._data_mean = cfg_data.MEAN
        self._data_std = cfg_data.STD
        self._use_bgr = cfg_dataset.BGR
        if self._split == 'train':
            self._jitter_min_scale = cfg_data.TRAIN_MIN_SCALES
            self._jitter_max_scale = cfg_data.TRAIN_MAX_SCALE
            self.random_horizontal_flip = cfg_data.RANDOM_FLIP
            self._use_color_augmentation = cfg_dataset.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg_dataset.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg_dataset.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg_dataset.TRAIN_PCA_EIGVEC
        else:
            self._jitter_min_scale = cfg_data.TEST_MIN_SCALES
            self._jitter_max_scale = cfg_data.TEST_MAX_SCALE
            self._test_force_flip = cfg_dataset.TEST_FORCE_FLIP
        
        self._fix_size = cfg_data.FIX_SIZE
    

    def shape_transform(self, imgs, boxes):
        if len(self._fix_size) == 0:
            # Short side to test_scale. Non-local and STRG uses 256.
            new_imgs, new_boxes = random_short_side_scale_jitter(
                imgs,
                min_sizes=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
                )  
        else:
            new_imgs, new_boxes = long_side_scale_and_pad(
                imgs,
                target_size=self._fix_size,
                boxes=boxes,
                padding='top'  # place image on top
            )
        return new_imgs, new_boxes


    def process(self, imgs, boxes=None):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.
        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.
        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        if boxes is not None:
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            boxes = clip_boxes_to_image(boxes, height, width)

            # `transform.py` is list of np.array. However, for AVA, we only have
            # one np.array.
            boxes = [boxes]

        # perform shape transform (rescaling and padding)
        imgs, boxes = self.shape_transform(imgs, boxes)

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        else:
            if self._test_force_flip:
                imgs, boxes = horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )

        # Convert image to CHW keeping BGR order.
        imgs = [HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)

        if boxes is not None:
            boxes = clip_boxes_to_image(
                boxes[0], imgs[0].shape[1], imgs[0].shape[2]
            )
        return imgs, boxes