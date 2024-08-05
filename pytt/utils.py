import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum, auto, unique
from typing import List, Tuple
from pathlib import Path

import cv2
import yaml
from munch import DefaultMunch
from PIL import Image
from pyaml_env import parse_config
from tqdm import tqdm
import imagesize


class DataFormat:
    __data_format_dict = {
        "VIDEO": [".mov", ".mp4", ".avi", ".mkv", ".wmv", ".flv", ".webm"],
        "IMAGE": [".jpg", ".png", "pgm", ".jpeg", ".tiff", ".tif", ".webp"],
        "CONFIG": [".txt", ".json"],
    }

    for key, value in __data_format_dict.items():
        value += list(map(lambda ext: ext.upper(), __data_format_dict[key]))
        __data_format_dict[key] = list(set(__data_format_dict[key]))

    VIDEO = __data_format_dict["VIDEO"]
    IMAGE = __data_format_dict["IMAGE"]
    CONFIG = __data_format_dict["CONFIG"]


def GetTToolsDir():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def UnTarFile(tar_file, output_dir):
    with tarfile.open(tar_file, "r") as tfile:
        tfile.extractall(output_dir)


def CheckPlatformInfo():
    system = platform.system()
    info = platform.uname()
    return info, system

# def IsImgFile(path):
    # return re.search(r'\.(jpg|png|jpeg|bmp)$', path, re.IGNORECASE)

def IsImgFile(path):
    if not os.path.isfile(file_path):
        raise RuntimeError(
            "input path is not valid file: {}".format(file_path))

    return bool(file_path.lower().endswith(tuple(DataFormat.IMAGE)))

def IsVideo(file_path):

    if not os.path.isfile(file_path):
        raise RuntimeError(
            "input path is not valid file: {}".format(file_path))

    return bool(file_path.lower().endswith(tuple(DataFormat.VIDEO)))


def NormalizeName(name):
    RE = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
    name = RE.sub('c', name)
    return "".join([x if x.isalnum() else "_" for x in name])


def NormPath(path):
    return os.path.abspath(os.path.expanduser(path))


def RunSubprocess(command, error_msg, cwd="./"):
    process_status = subprocess.call([command],
                                     shell=True,
                                     cwd=cwd,
                                     executable='/bin/bash')
    if process_status != 0:
        raise RuntimeError(error_msg)


def RunSubprocessCheckOutput(command, cwd=None):
    if type(command) == list:
        command = " ".join(command)
    process_status = subprocess.check_output(
        command, cwd=cwd, shell=True,
        executable='/bin/bash').decode('utf-8').split("\n")
    if process_status[-1] == "":
        process_status = process_status[:-1]
    process_status = GetRemapList(lambda x: x.strip(), process_status)
    return process_status


def RunSubprocessCMD(cmd, error_msg, cwd=None):
    RunSubprocessCMDList(cmd.split(), error_msg, cwd)


def RunSubprocessCMDList(cmd_list, error_msg, cwd=None, shell=True):
    if type(cmd_list) == list:
        cmd_list = " ".join(cmd_list)
    process_status = subprocess.run(cmd_list,
                                    cwd=cwd,
                                    shell=shell,
                                    executable='/bin/bash')
    if process_status.returncode != 0:
        raise RuntimeError(error_msg)


def ReplaceSpecialCharWithUnderscore(target_str):
    return target_str.translate(
        {ord(c): "_"
         for c in "!@#$%^&*()[]{};:,./<>?\ |`~-=+"})


def MkDir(target_dir, is_del_exist):
    if os.path.isdir(target_dir) and is_del_exist:
        shutil.rmtree(target_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def ListFilesPathInDirRecursive(folder_dir):

    return [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(folder_dir)
        for f in filenames if not f.startswith("._")
    ]


def ListDirInDirRecursive(folder_dir):

    return [
        os.path.join(dp, dir) for dp, dn, filenames in os.walk(folder_dir)
        for dir in dn if not dir.startswith("._")
    ]


def ListFilesInDir(file_dir, file_types_list):
    if not file_dir:
        return []
    file_types_tuple = tuple(file_types_list)
    file_dir_list = [
        f.path for f in os.scandir(file_dir) if f.is_file()
        and f.name.endswith(file_types_tuple) and not f.name.startswith("._")
    ]

    # assert file_dir_list != [], "#ERROR: cannot locate any files with the type within the given path %s" % file_dir
    file_dir_list.sort(key=lambda x: os.path.basename(x))
    return file_dir_list

def ListImageFilesInDir(file_dir):
    return ListFilesInDir(file_dir, DataFormat.IMAGE)


def ListFolderInDir(cluster_dir, is_skip_assert=False):
    folder_list = [
        f.path for f in os.scandir(cluster_dir)
        if f.is_dir() and not f.name.startswith(".")
    ]
    if not is_skip_assert:
        assert folder_list != [], "#ERROR: cannot locate any folders within the given root path %s" % cluster_dir
    folder_list.sort(key=lambda x: os.path.basename(x))
    return folder_list

def GetImageFolderProperty(image_folder_dir, img_ext : str, img_size : tuple):
    image_path_list = ListImageFilesInDir(image_folder_dir)
    if len(image_path_list) == 0:
        print(f"Not an image folder: {image_folder_dir}")
        return False

    first_img_size = imagesize.get(image_path_list[0])
    first_img_ext = Path(image_path_list[0]).suffix

    for image_path in image_path_list:
        img_size = imagesize.get(image_path)
        if img_size != first_img_size:
            print(f"Inconsistent image size: {image_path}")
            return False
        img_ext = Path(image_path).suffix
        if img_ext != first_img_ext:
            print(f"Inconsistent image ext: {img_ext}")
            return False

    return True


def GetAbsPathFilesInDir(path, datatype):
    filelist = [
        f.path for f in os.scandir(path) if not f.name.startswith('._')
        and f.is_file() and os.path.splitext(f.name)[1] in datatype
    ]
    filelist.sort()

    return filelist


def SaveList2File(output_file_path, content_list, mode="w"):
    assert os.path.exists(
        os.path.dirname(output_file_path)
    ), "#ERROR: given output path does not exists %s" % output_file_path
    with open(output_file_path, mode) as f:
        for line in content_list:
            f.write("%s\n" % line)


def LoadTxtFile2List(txt_file_path):
    assert os.path.isfile(
        txt_file_path), "#ERROR: cannot locate the given list file"
    with open(txt_file_path, "r") as f:
        txt_content_list = f.read().splitlines()
    return txt_content_list


def CSV2dict(csv_file, deli=","):
    csv_dict = {}
    with open(csv_file, "r") as f:
        for line in f.readlines():
            row = line.strip().split(deli)
            assert len(row) == 2 or len(
                row
            ) == 1, "#ERROR: incorrect format, now only support csv with 2 value on each row"
            if len(row) == 1:
                csv_dict[row[0].strip(" ")] = "-"
            else:
                csv_dict[row[0].strip(" ")] = row[1].strip()

    return csv_dict


def CSV2List(csv_file):
    with open(csv_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)
    return lines


def Dict2CSV(dictionary, csv_file):
    with open(csv_file, "w") as f:
        for key in dictionary.keys():
            f.write(",".join([key, dictionary[key]]) + "\n")


def GetVideoDuration(filename):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", filename
    ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    duration = float(result.stdout)
    assert duration > 0, "video duration = 0, please check your video {}".format(
        filename)
    return duration


def GetRemapList(lambda_func, iter_list):
    return list(map(lambda_func, iter_list))


def RsyncSubprocessCall(src,
                        dest,
                        is_complete_path,
                        is_dry_run,
                        error_msg,
                        archive=True,
                        verbose=True,
                        ignore_overwrite_condition=False,
                        ignore_existing=False,
                        remove_source_files=False,
                        prune_empty=True,
                        is_sshpass=False,
                        dest_delete=False,
                        include_file_type=[],
                        exclude_file_type=[]):

    rsync_command = "rsync"
    # archive mode; equals -rlptgoD (no -H,-A,-X)
    if archive:
        rsync_command += " -a"
    if verbose:
        rsync_command += " -v"
    if is_dry_run:
        rsync_command += " -n"
    if ignore_overwrite_condition:
        rsync_command += " -I"
    if ignore_existing:
        rsync_command += " --ignore-existing"
    if remove_source_files:
        rsync_command += " --remove-source-files"
    if prune_empty:
        rsync_command += " --prune-empty-dirs"
    if dest_delete:
        rsync_command += " --delete"
    if include_file_type:
        rsync_command += " --include */"
        for file_type in include_file_type:
            rsync_command += " --include=*%s" % file_type
        rsync_command += " --exclude=*"
    elif exclude_file_type:
        for file_type in exclude_file_type:
            rsync_command += " --exclude=*%s" % file_type

    if is_complete_path:
        src = src.rstrip("/") + "/"
        dest = dest.rstrip("/") + "/"
    rsync_command += " %s %s" % (src, dest)

    RunSubprocessCMD(rsync_command, error_msg)


def JoinOverlappingPath(path1, path2):
    return path1 + '/' + '/'.join(
        [i for i in path2.split('/') if i not in path1.split('/')])


def IsHomogeneousType(seq: list):
    if seq == []: return True
    iseq = iter(seq)
    first_type = type(next(iseq))
    return first_type if all((type(x) is first_type) for x in iseq) else False


def MS2HourMinuteSec(timeMS):
    timeSec = int(timeMS / 1000)
    hours = int(timeSec / (60 * 60))
    minutes = int((timeSec % (60 * 60)) / 60)
    seconds = (timeSec % 60)

    return str(hours).zfill(2) + ":" + str(minutes).zfill(2) + ":" + str(
        seconds).zfill(2)


def Gif2MP4(input_gif_file, output_mp4_file):
    import moviepy.editor as mp

    clip = mp.VideoFileClip(input_gif_file)
    clip.write_videofile(output_mp4_file)


def Image2JPG(input_picture_file, output_jpg_file):
    img = Image.open(input_picture_file).convert("RGB")
    img.save(output_jpg_file, "jpeg")


def GetVideoMetaInfoDictFromVideoFile(video_path):
    try:
        import ffmpeg
    except:
        print("use 'pip3 install ffmpeg-python' to use this function")
        raise RuntimeError(
            "ffmpeg-python not installed, use 'pip3 install ffmpeg-python'")
    video_streams = ffmpeg.probe(video_path, select_streams="v")

    return video_streams


def CheckValidString(string_to_check: str,
                     additional_allow_characters: str = "_") -> bool:
    """check if string have any invalid characters and return the test result in boolean

    :param str string_to_check: the string you want check
    :param str additional_allow_characters: you can add additional allow characters to check for, default to "_"

    :return:
        bool: the check result of the string, if pass return Ture, if not return False

    """
    return bool(
        re.match("^[A-Za-z0-9%s]*$" % additional_allow_characters,
                 string_to_check))


def FindWordInString(word_to_match, string_to_check) -> str:
    """ find the word to match in a string, and must match whole word so if
        match <ring> in <this is a sample string> wont work it would return None


    :param str word_to_match: word you want to search in string
    :param str string_to_check: string you want to seatch

    :return:
        <match object> if word is found
        None: if not word is found
    """

    return re.compile(r'\b({0})\b'.format(re.escape(word_to_match)),
                      flags=re.IGNORECASE).search(string_to_check)


def LoadJSON(json_file_path):
    with open(json_file_path, "r") as json_file:
        json_dict = json.load(json_file)

    return json_dict


def SaveJSON(json_file_path, json_dict, indent=4):
    with open(json_file_path, 'w') as json_file:
        json.dump(json_dict, json_file, indent=indent)
    print("saved to %s" % json_file_path)


def LoadYAML(yaml_file_path: str) -> DefaultMunch:
    # Can parse with env variables
    config = parse_config(yaml_file_path)
    config = DefaultMunch.fromDict(config)

    return config


def SaveYAML(yaml_file_path, yaml_dict):
    with open(yaml_file_path, "w") as yaml_file:
        yaml.safe_dump(yaml_dict, yaml_file)


def MultiProcessFunc(mp_target_list,
                     func,
                     desc="multiprocess_iter",
                     send_list_element_to_mp: bool = False,
                     func_args: Tuple = (),
                     func_kwargs: dict = {}) -> List:
    """ Multi-process interface function
    For example:
    Assume that we have a function called:
        crop_one_image(dataset_dir, class_id=0, labeler_id=9)
    
    Case 1:
    MultiProcessFunc(dataset_dir_list, crop_one_image, func_args=(0), func_kwargs={"labeler_id": 9}):
        This will call crop_one_image(0, labeler_id=9) in the loop.
    
    Case 2:
    MultiProcessFunc(dataset_dir_list, crop_one_image, send_list_element_to_mp=True, func_args=(0), func_kwargs={"labeler_id": 9}):
        This will call crop_one_image(dataset_dir, 0, labeler_id=9) in the loop.
    """

    with tqdm(total=len(mp_target_list),
              desc=desc) as progress, ProcessPoolExecutor() as pool:
        if send_list_element_to_mp:
            futures = [
                pool.submit(func, key, *func_args, **func_kwargs)
                for key in mp_target_list
            ]
        else:
            futures = [
                pool.submit(func, *func_args, **func_kwargs)
                for key in mp_target_list
            ]
        for _ in as_completed(futures):
            progress.update(1)
    return [f.result() for f in futures]
