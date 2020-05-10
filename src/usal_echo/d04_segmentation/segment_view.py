#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from optparse import OptionParser

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.misc import imresize
import hashlib
import datetime

from usal_echo import (
    model_dir,
    a4c_segmentation_model,
    a2c_segmentation_model
)

from usal_echo.d00_utils.log_utils import setup_logging
from usal_echo.d02_intermediate.download_dcm import dcm_to_segmentation_arrays
from usal_echo.d00_utils.db_utils import (
    dbReadWriteClean,
    dbReadWriteViews,
    dbReadWriteSegmentation
)
from usal_echo.d03_classification.evaluate_views import _groundtruth_views
from usal_echo.d04_segmentation.model_unet import Unet


logger = setup_logging(__name__, __name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def segmentChamber(videofile, dicomdir, view, model_path, seg_model, colour_scheme_lookup):
    """
    
    """
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
    modeldir = model_path

    if view == "a4c":
        g_1 = tf.Graph()
        with g_1.as_default():
            label_dim = 6  # a4c
            sess1 = tf.Session()
            model1 = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
            sess1.run(tf.local_variables_initializer())
            sess = sess1
            model = model1
        with g_1.as_default():
            saver = tf.train.Saver()
            saver.restore(
                sess1, os.path.join(modeldir, seg_model)
            )
    elif view == "a2c":
        g_2 = tf.Graph()
        with g_2.as_default():
            label_dim = 4
            sess2 = tf.Session()
            model2 = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
            sess2.run(tf.local_variables_initializer())
            sess = sess2
            model = model2
        with g_2.as_default():
            saver = tf.train.Saver()
            saver.restore(
                sess2, os.path.join(modeldir, seg_model)
            )

    outpath = "/home/ubuntu/data/04_segmentation/" + view + "/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    #look up colour scheme
    match_filename = 'a_' + str(videofile).split('_')[2].split('.')[0]
    instance_colour_scheme = colour_scheme_lookup[colour_scheme_lookup['filename'] == match_filename]['colour_scheme'].item()

    images, orig_images = dcm_to_segmentation_arrays(dicomdir, videofile, instance_colour_scheme)
    np_arrays_x3 = []
    images_uuid_x3 = []
    np_in_min = images.min()
    np_in_max = images.max()
    

    if view == "a4c":
        logger.info('predicitng a4c view')
        a4c_lv_segs, a4c_la_segs, a4c_lvo_segs, preds = extract_segs(
            images, orig_images, model, sess, 2, 4, 1
        )
        np_total = np.sum(a4c_lv_segs) + np.sum(a4c_la_segs) + np.sum(a4c_lvo_segs)
        np_arrays_x3.append(np.array(a4c_lv_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a4c_la_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a4c_lvo_segs).astype("uint8"))
        number_frames = (np.array(a4c_lvo_segs).astype("uint8").shape)[0]
        model_name = a4c_segmentation_model        
    if view == "a2c":
        logger.info('predicitng a2c view')
        a2c_lv_segs, a2c_la_segs, a2c_lvo_segs, preds = extract_segs(
            images, orig_images, model, sess, 2, 3, 1
        )
        np_total = np.sum(a2c_lv_segs) + np.sum(a2c_la_segs) + np.sum(a2c_lvo_segs)
        np_arrays_x3.append(np.array(a2c_lv_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a2c_la_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a2c_lvo_segs).astype("uint8"))
        number_frames = (np.array(a2c_lvo_segs).astype("uint8").shape)[0]
        model_name = a2c_segmentation_model
    j = 0    
    nrow = orig_images[0].shape[0]
    ncol = orig_images[0].shape[1]
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(imresize(preds, (nrow, ncol)))
    plt.savefig(outpath + "/" + videofile + "_" + str(j) + "_" + "segmentation.png")
    images_uuid_x3.append(
        hashlib.md5(
            (
                outpath + "/" + videofile + "_" + str(j) + "_" + "segmentation.png"
            ).encode()
        ).hexdigest()
    )
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(orig_images[0])
    plt.savefig(outpath + "/" + videofile + "_" + str(j) + "_" + "originalimage.png")
    images_uuid_x3.append(
        hashlib.md5(
            (
                outpath + "/" + videofile + "_" + str(j) + "_" + "originalimage.png"
            ).encode()
        ).hexdigest()
    )
    plt.close()
    background = Image.open(
        outpath + "/" + videofile + "_" + str(j) + "_" + "originalimage.png"
    )
    overlay = Image.open(
        outpath + "/" + videofile + "_" + str(j) + "_" + "segmentation.png"
    )
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    outImage = Image.blend(background, overlay, 0.5)
    outImage.save(outpath + "/" + videofile + "_" + str(j) + "_" + "overlay.png", "PNG")
    images_uuid_x3.append(
        hashlib.md5(
            (outpath + "/" + videofile + "_" + str(j) + "_" + "overlay.png").encode()
        ).hexdigest()
    )
    
    return [number_frames, model_name, np_arrays_x3, images_uuid_x3, np_in_min, np_in_max, np_total]



def segmentstudy(viewlist_a2c, viewlist_a4c, dcm_path, model_path, 
                 a4c_seg_model, a2c_seg_model):

    # set up for writing to segmentation schema
    io_views = dbReadWriteViews()
    io_segmentation = dbReadWriteSegmentation()

    column_names = [
        "study_id",
        "instance_id",
        "file_name",
        "num_frames",
        "model_name",
        "date_run",
        "output_np_lv",
        "output_np_la",
        "output_np_lvo",
        "output_image_seg",
        "output_image_orig",
        "output_image_overlay",
        "min_pixel_intensity",
        "max_pixel_intensity",
        "np_prediction_total"
        ]

    instances_unique_master_list = io_views.get_table("instances_unique_master_list")
    # below cleans the filename field to remove whitespace
    instances_unique_master_list["instancefilename"] = instances_unique_master_list[
        "instancefilename"
    ].apply(lambda x: str(x).strip())
    
    #below gets the colour scheme lookup
    #Get colour_scheme_lookup_table
    io_clean = dbReadWriteClean()
    colour_scheme_lookup = io_clean.get_table('colour_scheme_lookup')

    for video in viewlist_a4c:
        [number_frames, model_name, np_arrays_x3, images_uuid_x3, np_in_min, np_in_max, np_pred_total] = segmentChamber(
            video, dcm_path, "a4c", model_path, a4c_seg_model, colour_scheme_lookup
        )
        instancefilename = video.split("_")[2].split(".")[
            0
        ]  # split e.g. 'a_63712_45TXWHPP.dcm' to '45TXWHPP'
        studyidk = int(video.split("_")[1])
        # below filters to just the record of interest
        df = instances_unique_master_list.loc[
            (instances_unique_master_list["instancefilename"] == instancefilename)
            & (instances_unique_master_list["studyidk"] == studyidk)
        ]
        df = df.reset_index()
        instance_id = df.at[0, "instanceidk"]
        
        d = [
            studyidk,
            instance_id,
            str(video),
            number_frames,
            model_name,
            str(datetime.datetime.now()),
            np_arrays_x3[0],
            np_arrays_x3[1],
            np_arrays_x3[2],
            images_uuid_x3[0],
            images_uuid_x3[1],
            images_uuid_x3[2],
            np_in_min, 
            np_in_max,
            np_pred_total
            ]
        io_segmentation.save_prediction_numpy_array_to_db(d, column_names)
        logger.info('Saved an a4c predition')

    for video in viewlist_a2c:
        [number_frames, model_name, np_arrays_x3, images_uuid_x3, np_in_min, np_in_max, np_pred_total] = segmentChamber(
            video, dcm_path, "a2c", model_path, a2c_seg_model, colour_scheme_lookup
        )
        instancefilename = video.split("_")[2].split(".")[
            0
        ]  # split from 'a_63712_45TXWHPP.dcm' to '45TXWHPP'
        studyidk = int(video.split("_")[1])
        # below filters to just the record of interest
        df = instances_unique_master_list.loc[
            (instances_unique_master_list["instancefilename"] == instancefilename)
            & (instances_unique_master_list["studyidk"] == studyidk)
        ]
        df = df.reset_index()
        instance_id = df.at[0, "instanceidk"]
        d = [
            studyidk,
            instance_id,
            str(video),
            number_frames,
            model_name,
            str(datetime.datetime.now()),
            np_arrays_x3[0],
            np_arrays_x3[1],
            np_arrays_x3[2],
            images_uuid_x3[0],
            images_uuid_x3[1],
            images_uuid_x3[2],
            np_in_min, 
            np_in_max,
            np_pred_total
            ]
            
        io_segmentation.save_prediction_numpy_array_to_db(d, column_names)
        logger.info('Saved an a2c predition')
    
    return 1


def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output


def extract_segs(images, orig_images, model, sess, lv_label, la_label, lvo_label):
    segs = []
    preds = np.argmax(model.predict(sess, images[0:1])[0, :, :, :], 2)
    label_all = list(range(1, 8))
    label_good = [lv_label, la_label, lvo_label]
    for i in label_all:
        if not i in label_good:
            preds[preds == i] = 0
    for i in range(len(images)):
        seg = np.argmax(model.predict(sess, images[i : i + 1])[0, :, :, :], 2)
        segs.append(seg)
    lv_segs = []
    lvo_segs = []
    la_segs = []
    for seg in segs:
        la_seg = create_seg(seg, la_label)
        lvo_seg = create_seg(seg, lvo_label)
        lv_seg = create_seg(seg, lv_label)
        lv_segs.append(lv_seg)
        lvo_segs.append(lvo_seg)
        la_segs.append(la_seg)
    return lv_segs, la_segs, lvo_segs, preds


def run_segment(
    dcm_path,
    model_path,
    img_dir,
    classification_model_name,
    a4c_seg_model, a2c_seg_model,
    date_run=datetime.date.today(),
):

    path = dcm_path

    file_path = []
    filenames = []

    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith("dcm_raw"):
                file_path.append(os.path.join(r, file))
                fullfilename = os.path.basename(os.path.join(r, file))
                filenames.append(str(fullfilename).split(".")[0])

    logger.info("Number of files in the directory: {}".format(len(file_path)))
    filename_df = pd.DataFrame(filenames)

    predict_truth = _groundtruth_views()

    predictions_df = predict_truth[
        (predict_truth["img_dir"] == img_dir)
        & (predict_truth["model_name"] == classification_model_name)
        & (pd.to_datetime(predict_truth["date_run"]).dt.date == date_run)
    ]

    file_predictions = pd.merge(
        filename_df, predictions_df, how="inner", left_on=[0], right_on=["file_name"]
    )

    logger.info(
        "Number of files successfully matched with classification predictions: {}".format(
            file_predictions.shape[0]
        )
    )

    start = time.time()

    viewlist_a4c = file_predictions[file_predictions["view4_seg"] == "a4c"]["file_name"]
    viewlist_a4c = viewlist_a4c.apply(lambda x: x + ".dcm")
    viewlist_a4c = viewlist_a4c.to_list()
    logger.info("{} a4c files added to the view list".format(len(viewlist_a4c)))

    viewlist_a2c = file_predictions[file_predictions["view4_seg"] == "a2c"]["file_name"]
    viewlist_a2c = viewlist_a2c.apply(lambda x: x + ".dcm")
    viewlist_a2c = viewlist_a2c.to_list()
    logger.info("{} a2c files added to the view list".format(len(viewlist_a2c)))

    segmentstudy(viewlist_a2c, viewlist_a4c, dcm_path, model_path, a4c_seg_model, a2c_seg_model)
    end = time.time()
    viewlist = viewlist_a2c + viewlist_a4c
    logger.info(
        "time:  " + str(end - start) + " seconds for " + str(len(viewlist)) + " videos"
    )
