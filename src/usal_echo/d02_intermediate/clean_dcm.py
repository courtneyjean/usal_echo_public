#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 4 14:32:40 2019

@author: wiebket
"""

import pandas as pd
from json import load
import os
from pathlib import Path

from usal_echo import usr_dir
from usal_echo.d00_utils.db_utils import dbReadWriteRaw, dbReadWriteClean
from usal_echo.d00_utils.log_utils import setup_logging

logger = setup_logging(__name__, __name__)

dcm_tags = os.path.join(usr_dir, "conf", "dicom_tags.json")


def clean_dcm_meta():
    """Selects a subset of dicom metadata tags and saves them to postgres.
    
    **Requirements:
    json formatted config file with dicom tag descriptions and values 
    in d02_intermediate/dicom_tags.json

    """
    with open(dcm_tags) as f:
        dicom_tags = load(f)
    for k, v in dicom_tags.items():
        dicom_tags[k] = tuple(v)

    io_raw = dbReadWriteRaw()
    io_clean = dbReadWriteClean()
    metadata = io_raw.get_table("metadata")

    metadata["tags"] = list(zip(metadata["tag1"], metadata["tag2"]))
    meta_lite = metadata[metadata["tags"].isin(dicom_tags.values())]

    io_clean.save_to_db(meta_lite, "meta_lite")
    
        #create a colour scheme lookup
    #Create a colour scheme lookup for filenames
    colour_scheme_lookup =  meta_lite[(meta_lite['tag1'] == '0028') & (meta_lite['tag2'] == '0004')].copy()
    colour_scheme_lookup =  colour_scheme_lookup.drop_duplicates()
    colour_scheme_lookup =  colour_scheme_lookup.drop_duplicates(subset='filename', keep='first')
    colour_scheme_lookup =  colour_scheme_lookup.rename(columns={'value':'colour_scheme'})
    
    io_clean.save_to_db(colour_scheme_lookup, "colour_scheme_lookup")
    
    logger.info("Metadata filtered.")
