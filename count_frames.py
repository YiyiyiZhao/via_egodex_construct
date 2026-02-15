#!/usr/bin/env python3
"""
统计 EgoDex frames 目录中的 jpg 文件数量
- 对于 all_descriptions.json 中的 video_name，统计对应目录的 jpg 文件数
- 对于不在 all_descriptions.json 中的 frames 目录，也统计 jpg 文件数
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# 配置路径
DESCRIPTIONS_FILE = "descriptions/all_descriptions.json"
FRAMES_BASE_DIR = "/data1/zy/full_data/EgoDex/frames"

def count_jpg_files(directory):
    """统计目录下的 jpg 文件数量"""
    if not os.path.exists(directory):
        return 0

    jpg_files = list(Path(directory).glob("*.jpg"))
    return len(jpg_files)

def main():
    # 读取 all_descriptions.json
    print(f"读取 {DESCRIPTIONS_FILE}...")
    with open(DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    # 提取所有 video_name
    video_names_in_json = set()
    for item in descriptions:
        video_names_in_json.add(item['video_name'])

    print(f"在 all_descriptions.json 中找到 {len(video_names_in_json)} 个视频")
    print(f"\n{'='*80}")
    print("统计 all_descriptions.json 中的视频帧数：")
    print(f"{'='*80}")

    # 统计 all_descriptions.json 中的视频
    results_in_json = {}
    for video_name in sorted(video_names_in_json):
        video_dir = os.path.join(FRAMES_BASE_DIR, video_name)
        jpg_count = count_jpg_files(video_dir)
        results_in_json[video_name] = jpg_count
        print(f"{video_name}: {jpg_count} 张 jpg")

    # 获取 frames 目录下所有的子目录
    print(f"\n{'='*80}")
    print("检查 frames 目录下所有子目录...")
    print(f"{'='*80}")

    if os.path.exists(FRAMES_BASE_DIR):
        all_frame_dirs = set()
        for item in os.listdir(FRAMES_BASE_DIR):
            item_path = os.path.join(FRAMES_BASE_DIR, item)
            if os.path.isdir(item_path):
                all_frame_dirs.add(item)

        # 找出不在 all_descriptions.json 中的目录
        dirs_not_in_json = all_frame_dirs - video_names_in_json

        if dirs_not_in_json:
            print(f"\n找到 {len(dirs_not_in_json)} 个不在 all_descriptions.json 中的目录：")
            print(f"{'='*80}")

            results_not_in_json = {}
            for dir_name in sorted(dirs_not_in_json):
                video_dir = os.path.join(FRAMES_BASE_DIR, dir_name)
                jpg_count = count_jpg_files(video_dir)
                results_not_in_json[dir_name] = jpg_count
                print(f"{dir_name}: {jpg_count} 张 jpg")
        else:
            print("\n所有 frames 目录都在 all_descriptions.json 中")
    else:
        print(f"\n警告: frames 目录不存在: {FRAMES_BASE_DIR}")

    # 输出统计摘要
    print(f"\n{'='*80}")
    print("统计摘要：")
    print(f"{'='*80}")
    print(f"all_descriptions.json 中的视频数量: {len(results_in_json)}")
    print(f"all_descriptions.json 中的总帧数: {sum(results_in_json.values())}")

    if dirs_not_in_json:
        print(f"不在 all_descriptions.json 中的目录数量: {len(results_not_in_json)}")
        print(f"不在 all_descriptions.json 中的总帧数: {sum(results_not_in_json.values())}")

    # 检查是否有目录不存在或为空
    empty_dirs = [name for name, count in results_in_json.items() if count == 0]
    if empty_dirs:
        print(f"\n{'='*80}")
        print(f"警告: 以下 {len(empty_dirs)} 个目录不存在或为空:")
        print(f"{'='*80}")
        for dir_name in empty_dirs:
            print(f"  - {dir_name}")

if __name__ == "__main__":
    main()
