import os
import re

def extract_record_number(filename):
    """Extract MIT-BIH record number (digits before underscore)."""
    m = re.match(r"(\d+)_", filename)
    return int(m.group(1)) if m else None

def build_record_dicts(data_folder):
    """Return {record_number: filename} for ECG and annotation files."""
    files = os.listdir(data_folder)

    ekg_files = [f for f in files if re.search(r"_ekg\.csv$", f)]
    ann_files = [f for f in files if re.search(r"_annotations_1\.csv$", f)]

    ekg_dict = {extract_record_number(f): f for f in ekg_files}
    ann_dict = {extract_record_number(f): f for f in ann_files}

    return ekg_dict, ann_dict