#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Aug  2 13:46:14 2019

@author: court
"""
import os
import numpy as np
from medpy.metric.binary import dc, jc, hd, precision, recall, sensitivity, specificity

from usal_echo.d00_utils.db_utils import dbReadWriteClean, dbReadWriteSegmentation
from usal_echo.d00_utils.log_utils import setup_logging

logger = setup_logging(__name__, __name__)


def evaluate_masks(dcm_dir_path):
    # Go through the ground truth table and write IOUS

    # Prediction Table: "instance_id","study_id", "view_name", "frame", "output_np_lv", "output_np_la",
    #        "output_np_lvo","output_image_seg", "output_image_orig", "output_image_overlay", "date_run",
    #        "file_name"
    # Ground truth table: ground_truth_id, instance_id, frame, chamber, study_id, view_name, numpy_array
    # Evaluation Table: evaluation_id, instance_id, frame, chamber, study_id, score_type, score_value
    
    #just run eval for dcm_dir_path
    path = dcm_dir_path
    dataset_name = str(dcm_dir_path).split('/')[-1]

    file_path = []
    filenames = []

    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith("dcm_raw"):
                file_path.append(os.path.join(r, file))
                fullfilename = os.path.basename(os.path.join(r, file))
                filenames.append(str(fullfilename).split(".")[0].split("_")[-1])

    logger.info("Number of files in the directory: {}".format(len(file_path)))

    io_segmentation = dbReadWriteSegmentation()
    ground_truths = io_segmentation.get_segmentation_table("ground_truths")
    
    #match ground truth with filenames
    ground_truth_files = ground_truths[ground_truths['file_name'].isin(filenames)]
    
    #get table of voxel spacing values
    voxel_spacing_df = get_voxel_spacing_for_instances(ground_truth_files)
    #set index to be filename
    voxel_spacing_df = voxel_spacing_df.set_index('file_name')

    # Go through the ground truth table and write IOUS, DICE and Hausdorff distance

    for index, gt in ground_truth_files.iterrows(): #only run for files in directory!
        # match the gt to the prediction table
        gt_instance_id = gt["instance_id"]
        gt_study_id = gt["study_id"]
        gt_chamber = gt["chamber"]
        gt_view_name = gt["view_name"]
        gt_frame_no = gt["frame"]
        gt_file_name = gt["file_name"]
        
        
        #take the min of x or y scale spacing (appear to be the same for all files)
        try:
            voxel_spacing = float(voxel_spacing_df.loc[gt['file_name']]['value'])
        except TypeError:
            logger.info('voxel spacing can not be converted to a float')
            voxel_spacing = 0.013

        #min distance of 0.012
        if voxel_spacing > 0.012:
            pass
        else:
            voxel_spacing = 0.013
            
        pred = io_segmentation.get_instance_from_segementation_table(
            "predictions", gt_instance_id
        )
        pred = pred.reset_index()
        logger.info(
            "got {} predictions details for instance {}".format(
                len(pred), gt_instance_id
            )
        )

        if len(pred.index) > 0:
            pred_last = pred.head(1)
            pred_view_name = gt["view_name"]
            pred_seg_model = pred_last['model_name']
            # retrieve gt numpy array
            gt_numpy_array = io_segmentation.convert_to_np(
                gt["numpy_array"], 1
            )  # frame = 1, as it wants number of frames in np array, not frame number
            if gt_chamber == "la":
                pred_numpy_array = io_segmentation.convert_to_np(
                    pred["output_np_la"][0], pred["num_frames"][0]
                )
            elif gt_chamber == "lv":
                pred_numpy_array = io_segmentation.convert_to_np(
                    pred["output_np_lv"][0], pred["num_frames"][0]
                )
            elif gt_chamber == "lvo":
                pred_numpy_array = io_segmentation.convert_to_np(
                    pred["output_np_lvo"][0], pred["num_frames"][0]
                )
            else:
                logger.error("invalid chamber")

            # get the frame of the prediction, that corresponds to the frame of the ground thruth
            pred_numpy_array_frame = pred_numpy_array[gt_frame_no, :, :]

            # calculate measures
            reported_iou = jc(gt_numpy_array, pred_numpy_array_frame)            
            reported_dice = dc(gt_numpy_array, pred_numpy_array_frame)    
            reported_precision = precision(gt_numpy_array, pred_numpy_array_frame)
            reported_recall = recall(gt_numpy_array, pred_numpy_array_frame)
            reported_sensitivity = sensitivity(gt_numpy_array, pred_numpy_array_frame)
            reported_specificity = specificity(gt_numpy_array, pred_numpy_array_frame)
            
            zhang_dice = zhang_modified_dice(gt_numpy_array, pred_numpy_array_frame)
            
            try:
                reported_hausdorff = hd(gt_numpy_array, pred_numpy_array_frame, voxelspacing=voxel_spacing)
            except:
                reported_hausdorff = 0;
                logger.error('hausdorf distance function fails when array equals zero')    
            
            # write evaluation metrics to db
            # Evaluation Table: evaluation_id, instance_id, frame, chamber, study_id, score_type, score_value
            d_columns = [
                "instance_id",
                "frame",
                "file_name",
                "chamber",
                "study_id",
                "score_type",
                "score_value",
                "gt_view_name",
                "pred_view_name",
                "dataset",
                "model_name"]
            
            metric_list = {"Jaccard": reported_iou
                           , "Dice": reported_dice
                           , "Hausdorff": reported_hausdorff
                           , "Precison" : reported_precision
                           , "Recall" : reported_recall
                           , "Sensitivity" : reported_sensitivity
                           , "Specificity": reported_specificity
                           , "Zhang modified dice": zhang_dice
                           }
            
            for label, value in metric_list.items():
                d = [gt_instance_id
                     , gt["frame"]
                     , gt_file_name
                     , gt_chamber
                     , gt_study_id
                     , label
                     , value
                     , gt_view_name
                     , pred_view_name
                     , dataset_name
                     , pred_seg_model]
                io_segmentation.save_seg_evaluation_to_db(d, d_columns)
                logger.info("{} metric record, with value of {}".format(label, value))
        
        else:
            logger.error(
                "No record exists for study id {} & instance id {}".format(
                    gt_study_id, gt_instance_id
                )
            )

def zhang_modified_dice(gt, pred): #, seg):
    #gt_seg = create_seg(gt, seg)
    #pred_seg = create_seg(pred, seg)
    overlap = np.minimum(gt, pred)
    
    return 2*np.sum(overlap)/(np.sum(gt) + np.sum(pred))

def get_voxel_spacing_for_instances(df):
    io_clean = dbReadWriteClean()
    df_dcm = io_clean.get_table("meta_lite") #organised by filename
    
    df_dcm.rename(columns={"filename": "file_name"}, inplace=True)
    df_dcm["file_name"] = df_dcm["file_name"].str.rstrip()
    df_dcm["file_name"] = df_dcm["file_name"].str.replace("a_", "") #cut the a_ from the string
    df_dcm["value"] = df_dcm["value"].str.replace(".", "0.") 
        
    df_dcm["tag1"] = df_dcm["tag1"].astype(str)  # consistency with tag2
    voxel_spacing_df = df_dcm.loc[(df_dcm["tag1"] == "18") & (df_dcm["tag2"] == "602c")] #just return x spacing
    
    return voxel_spacing_df   
