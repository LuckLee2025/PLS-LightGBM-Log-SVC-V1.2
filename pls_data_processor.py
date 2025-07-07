# -*- coding:utf-8 -*-
"""
排列三数据处理器
================

本脚本负责从网络上获取排列三的历史开奖数据，并将其保存为标准化的CSV文件。

主要功能:
1.  从文本文件 (pl3_asc.txt) 获取排列三历史数据。
2.  数据清理和格式化，确保数据质量。
3.  将数据保存到CSV文件 ('pls.csv') 中。
4.  具备良好的错误处理和日志记录能力。

数据格式: Seq, red_1, red_2, red_3
排列三规则: 3个数字，每个数字范围0-9，有顺序，允许重复
"""

import os
import requests
import pandas as pd
import sys
import logging
from datetime import datetime

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 获取脚本所在的目录，确保路径的相对性
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 目标CSV文件的完整路径
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'pls.csv')

# 网络数据源URL - 排列三数据源
TXT_DATA_URL = 'https://data.17500.cn/pl3_asc.txt'

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('pls_data_processor')


# ==============================================================================
# --- 数据获取与处理模块 ---
# ==============================================================================

def fetch_pl3_data():
    """
    获取排列3数据并保存为pls.csv，按Seq升序，去掉日期列
    
    基于提供的参考代码，结合现有项目的错误处理机制
    """
    logger.info("开始获取排列三数据...")
    
    # 创建保存目录（如果不存在）
    save_directory = SCRIPT_DIR
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:
        # 发送HTTP GET请求获取数据
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(TXT_DATA_URL, headers=headers, timeout=30)
        response.raise_for_status()  # 检查请求是否成功
        response.encoding = 'utf-8'  # 显式设置编码
        logger.info(f"成功从 {TXT_DATA_URL} 获取数据")
    except requests.RequestException as e:
        logger.error(f"请求数据时出错: {e}")
        return

    data = []
    lines = response.text.strip().split('\n')
    logger.info(f"获取到 {len(lines)} 行原始数据")

    for line_num, line in enumerate(lines, 1):
        if len(line) < 10:
            continue  # 跳过无效行

        try:
            # 仅分割第一个逗号，忽略后续数据
            parts = line.split(',', 1)
            if not parts:
                continue  # 跳过空行

            first_part = parts[0].strip()
            fields = first_part.split()

            # 如果数据包含日期，fields 应该至少有 4 个字段（Seq + 日期 + 3个红球）
            # 如果不包含日期，fields 应该至少有 4 个字段（Seq + 3个红球）
            if len(fields) < 4:
                logger.debug(f"跳过行{line_num}（字段不足4个）：{line}")
                continue

            seq = fields[0]
            
            # 判断第二个字段是否为日期格式（简单判断是否包含'-'）
            if '-' in fields[1]:
                # 数据包含日期，红球从 fields[2] 开始
                if len(fields) < 5:
                    logger.debug(f"跳过行{line_num}（红球数量不足3个）：{line}")
                    continue
                red_balls = fields[2:5]  # 提取3个红球
            else:
                # 数据不包含日期，红球从 fields[1] 开始
                red_balls = fields[1:4]
                if len(red_balls) != 3:
                    logger.debug(f"跳过行{line_num}（红球数量不足3个）：{line}")
                    continue

            # 检查红球数量是否正确
            if len(red_balls) != 3:
                logger.debug(f"跳过行{line_num}（红球数量不足3个）：{line}")
                continue

            # 验证数字范围 (0-9)
            try:
                red_nums = [int(ball) for ball in red_balls]
                if not all(0 <= num <= 9 for num in red_nums):
                    logger.warning(f"跳过行{line_num}（数字超出0-9范围）：{line}")
                    continue
            except ValueError:
                logger.warning(f"跳过行{line_num}（数字格式错误）：{line}")
                continue

            # 构建数据字典
            item = {'Seq': seq}
            for i in range(1, 4):
                item[f'red_{i}'] = red_balls[i-1]

            data.append(item)

        except Exception as e:
            logger.warning(f"解析第{line_num}行时出错: {e}，跳过此行")
            continue

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['Seq'] + [f'red_{i}' for i in range(1, 4)])

    if df.empty:
        logger.error("没有提取到任何数据。请检查数据格式或数据源是否可用。")
        return

    logger.info(f"成功解析 {len(df)} 条有效记录")

    # 将Seq转换为整数以便排序
    try:
        df['Seq'] = df['Seq'].astype(int)
    except ValueError as e:
        logger.error(f"转换Seq为整数时出错: {e}")
        return

    # 按Seq升序排序
    df.sort_values(by='Seq', inplace=True)
    logger.info("数据按期号升序排列完成")

    try:
        # 保存为CSV文件
        df.to_csv(CSV_FILE_PATH, encoding="utf-8", index=False)
        logger.info(f"数据已成功保存到 {CSV_FILE_PATH}")
        logger.info(f"数据范围: 第{df['Seq'].min()}期 - 第{df['Seq'].max()}期")
    except Exception as e:
        logger.error(f"保存数据时出错: {e}")


def load_existing_data():
    """加载现有的CSV数据"""
    if os.path.exists(CSV_FILE_PATH):
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            logger.info(f"加载现有数据: {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"加载现有数据失败: {e}")
    return None


def update_data():
    """更新数据：获取最新数据并与现有数据合并"""
    logger.info("开始数据更新流程...")
    
    # 备份现有文件
    if os.path.exists(CSV_FILE_PATH):
        backup_path = CSV_FILE_PATH.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            import shutil
            shutil.copy2(CSV_FILE_PATH, backup_path)
            logger.info(f"已备份现有数据到: {backup_path}")
        except Exception as e:
            logger.warning(f"备份文件失败: {e}")
    
    # 获取新数据
    fetch_pl3_data()


if __name__ == "__main__":
    # 如果是直接运行，则执行数据更新
    update_data() 