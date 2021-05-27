from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.face_utils import shape_to_np
from imutils.face_utils import visualize_facial_landmarks
import imutils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2_plt_imshow import cv2_plt_imshow, plt_format
import os


def bb_to_rect(bb):
    left = bb[0]  # startX
    top = bb[1]  # startY
    right = bb[2]  # endX
    bottom = bb[3]  # endY
    dlibRect = dlib.rectangle(left, top, right, bottom)
    return dlibRect


def visualize_points(im, shape, skip=False):
    # cv2.rectangle(faceAligned, (x, y), (x + w, y + h), (0, 255, 0), 2)  # bounding box of detection
    if skip:
        face_circle = im
    else:
        for (x, y) in shape:  # draw the landmarks points in the detected face
            face_circle = cv2.circle(im, (x, y), 3, (252, 126, 255), -1)
    return face_circle


def dlib_aligner(image2, predictor, bb, size):
    fa = FaceAligner(predictor, desiredFaceWidth=size)
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    rect = bb_to_rect(bb)  # from bounding boxes to dlib rectangule format
    faceAligned = fa.align(image2, gray, rect)
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    face_draw = visualize_facial_landmarks(image2, shape)
    face_circle = visualize_points(image2, shape)
    return faceAligned, face_draw, face_circle


def add_titlebox(ax, text, nolabels):
    ax.text(.55, .8, text,
    position=(0.03, 0.05),
    transform=ax.transAxes,
    bbox=dict(facecolor='white', alpha=0.5),
    fontsize=11.5)
    ax.tick_params(axis='both', which='both', **nolabels)
    return ax


def toshow(name, face_align, face_draw, face_circle, face_resnet, i, save_emb=True):
    cv2_plt_imshow(face_align)
    cv2_plt_imshow(face_draw)
    cv2_plt_imshow(face_circle)
    cv2_plt_imshow(face_resnet)
    sides = ('left', 'right', 'top', 'bottom')
    nolabels = {s: False for s in sides}
    nolabels.update({'label%s' % s: False for s in sides})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(name, fontsize=16)
    axs[0, 0].imshow(plt_format(face_resnet), interpolation='nearest', aspect='auto')
    axs[0, 1].imshow(plt_format(face_align), interpolation='nearest', aspect='auto')
    # axs[0].set_title(name, fontsize=16)
    axs[1, 0].imshow(plt_format(face_circle), interpolation='nearest', aspect='auto')
    axs[1, 1].imshow(plt_format(face_draw), interpolation='nearest', aspect='auto')
    add_titlebox(axs[0, 1], 'Aligned Face', nolabels)
    add_titlebox(axs[1, 1], 'Facial Landmarks', nolabels)
    add_titlebox(axs[1, 0], 'Landmark Points', nolabels)  # Landmark Points
    add_titlebox(axs[0, 0], 'Resnet Detection', nolabels)
    fig.tight_layout()

    if save_emb is True:
        fig.savefig(os.path.join('models', 'embeddings', name + '_' + str(i+1) + '.png'))
    else:
        fig.show()
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

    plt.close('all')


def toshow_128(face_align, vec):
    pass
