# -*- coding: utf-8 -*-
"""
排列三推荐结果验证与奖金计算器
=================================

本脚本旨在自动评估 `pls_analyzer.py` 生成的推荐号码的实际表现。

工作流程:
1.  读取 `pls.csv` 文件，获取所有历史开奖数据。
2.  确定最新的一期为"评估期"，倒数第二期为"报告数据截止期"。
3.  根据"报告数据截止期"，在当前目录下查找对应的分析报告文件
    (pls_analysis_output_*.txt)。
4.  从找到的报告中解析出推荐的排列三号码。
5.  使用"评估期"的实际开奖号码，核对所有推荐投注的中奖情况。
6.  计算总奖金，并将详细的中奖结果追加记录到主报告文件 
    `latest_pls_calculation.txt` 中。
"""

import os
import re
import glob
import csv
from datetime import datetime
import traceback
from typing import Optional, Tuple, List, Dict

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 脚本需要查找的分析报告文件名的模式
REPORT_PATTERN = "pls_analysis_output_*.txt"
# 开奖数据源CSV文件
CSV_FILE = "pls.csv"
# 最终生成的主评估报告文件名
MAIN_REPORT_FILE = "latest_pls_calculation.txt"

# 主报告文件中保留的最大记录数
MAX_NORMAL_RECORDS = 10  # 保留最近10次评估
MAX_ERROR_LOGS = 20      # 保留最近20条错误日志

# 排列三奖金对照表 (元)
PRIZE_VALUES = {
    "直选": 1040,    # 直选奖金：所选号码与中奖号码相同且顺序一致
    "组选3": 346,    # 组选3奖金：中奖号码中任意两位数字相同，所选号码与中奖号码相同且顺序不限
    "组选6": 173,    # 组选6奖金：所选号码与中奖号码相同且顺序不限
}

# ==============================================================================
# --- 工具函数 ---
# ==============================================================================

def log_message(message: str, level: str = "INFO"):
    """一个简单的日志打印函数，用于在控制台显示脚本执行状态。"""
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')} - {message}")

def robust_file_read(file_path: str) -> Optional[str]:
    """
    一个健壮的文件读取函数，能自动尝试多种编码格式。

    Args:
        file_path (str): 待读取文件的路径。

    Returns:
        Optional[str]: 文件内容字符串，如果失败则返回 None。
    """
    if not os.path.exists(file_path):
        log_message(f"文件未找到: {file_path}", "ERROR")
        return None
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, IOError):
            continue
    log_message(f"无法使用任何支持的编码打开文件: {file_path}", "ERROR")
    return None

# ==============================================================================
# --- 数据解析与查找模块 ---
# ==============================================================================

def get_period_data_from_csv(csv_content: str) -> Tuple[Optional[Dict], Optional[List]]:
    """
    从CSV文件内容中解析出所有期号的开奖数据。

    Args:
        csv_content (str): 从CSV文件读取的字符串内容。

    Returns:
        Tuple[Optional[Dict], Optional[List]]:
            - 一个以期号为键，开奖数据为值的字典。
            - 一个按升序排序的期号列表。
            如果解析失败则返回 (None, None)。
    """
    if not csv_content:
        log_message("输入的CSV内容为空。", "WARNING")
        return None, None
    period_map, periods_list = {}, []
    try:
        reader = csv.reader(csv_content.splitlines())
        next(reader)  # 跳过表头
        for i, row in enumerate(reader):
            if len(row) >= 4 and re.match(r'^\d{4,7}$', row[0]):
                try:
                    period, red_1, red_2, red_3 = row[0], int(row[1]), int(row[2]), int(row[3])
                    # 验证数字范围
                    if not all(0 <= num <= 9 for num in [red_1, red_2, red_3]):
                        continue
                    period_map[period] = {'numbers': [red_1, red_2, red_3]}
                    periods_list.append(period)
                except (ValueError, IndexError):
                    log_message(f"CSV文件第 {i+2} 行数据格式无效，已跳过: {row}", "WARNING")
    except Exception as e:
        log_message(f"解析CSV数据时发生严重错误: {e}", "ERROR")
        return None, None
    
    if not period_map:
        log_message("未能从CSV中解析到任何有效的开奖数据。", "WARNING")
        return None, None
        
    return period_map, sorted(periods_list, key=int)

def find_matching_report(target_period: str) -> Optional[str]:
    """
    在当前目录查找其数据截止期与 `target_period` 匹配的最新分析报告。

    Args:
        target_period (str): 目标报告的数据截止期号。

    Returns:
        Optional[str]: 找到的报告文件的路径，如果未找到则返回 None。
    """
    log_message(f"正在查找数据截止期为 {target_period} 的分析报告...")
    candidates = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file_path in glob.glob(os.path.join(script_dir, REPORT_PATTERN)):
        content = robust_file_read(file_path)
        if not content: continue
        
        match = re.search(r'分析基于数据:\s*截至\s*(\d+)\s*期', content)
        if match and match.group(1) == target_period:
            try:
                timestamp_str_match = re.search(r'_(\d{8}_\d{6})\.txt$', file_path)
                if timestamp_str_match:
                    timestamp_str = timestamp_str_match.group(1)
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidates.append((timestamp, file_path))
            except (AttributeError, ValueError):
                continue
    
    if not candidates:
        log_message(f"未找到数据截止期为 {target_period} 的分析报告。", "WARNING")
        return None
        
    candidates.sort(reverse=True)
    latest_report = candidates[0][1]
    log_message(f"找到匹配的最新报告: {os.path.basename(latest_report)}", "INFO")
    return latest_report

def parse_recommendations_from_report(content: str) -> Dict:
    """
    从分析报告内容中解析出排列三推荐号码（单式和复式）。

    Args:
        content (str): 分析报告的文本内容。

    Returns:
        Dict: 包含单式推荐和复式推荐的字典
        {
            'single': List[List[int]],  # 单式推荐
            'duplex': Dict,             # 复式推荐
            'target_period': str        # 目标期号
        }
    """
    result = {
        'single': [],
        'duplex': {},
        'target_period': ''
    }
    
    # 解析目标期号
    target_match = re.search(r'本次预测目标:\s*第\s*(\d+)\s*期', content)
    if target_match:
        result['target_period'] = target_match.group(1)
    
    # 解析单式推荐号码
    rec_pattern = re.compile(r'注\s*\d+:\s*\[([0-9\s,]+)\]')
    for match in rec_pattern.finditer(content):
        try:
            numbers_str = match.group(1)
            numbers = [int(x.strip()) for x in re.findall(r'\d', numbers_str)]
            if len(numbers) == 3 and all(0 <= num <= 9 for num in numbers):
                result['single'].append(numbers)
        except ValueError:
            continue
    
    # 解析复式推荐号码
    duplex_patterns = {
        '百位': re.compile(r'百位\s*\(Top\s*\d+\):\s*([0-9\s]+)'),
        '十位': re.compile(r'十位\s*\(Top\s*\d+\):\s*([0-9\s]+)'),
        '个位': re.compile(r'个位\s*\(Top\s*\d+\):\s*([0-9\s]+)')
    }
    
    for position, pattern in duplex_patterns.items():
        match = pattern.search(content)
        if match:
            try:
                numbers_str = match.group(1)
                numbers = [int(x.strip()) for x in numbers_str.split() if x.strip().isdigit()]
                numbers = [num for num in numbers if 0 <= num <= 9]
                if numbers:
                    result['duplex'][position] = numbers
            except ValueError:
                continue
    
    log_message(f"从报告中解析出 {len(result['single'])} 个单式推荐号码")
    if result['duplex']:
        duplex_info = ', '.join([f"{pos}: {nums}" for pos, nums in result['duplex'].items()])
        log_message(f"从报告中解析出复式推荐: {duplex_info}")
    
    return result

def calculate_prize(recommendations: List[List[int]], prize_numbers: List[int]) -> Tuple[int, Dict, List]:
    """
    计算排列三推荐号码的中奖情况和总奖金

    Args:
        recommendations: 推荐号码列表
        prize_numbers: 开奖号码 [百位, 十位, 个位]

    Returns:
        Tuple[int, Dict, List]: (总奖金, 奖级统计, 中奖详情)
    """
    total_prize = 0
    prize_counts = {}
    winning_details = []
    
    for i, rec_numbers in enumerate(recommendations):
        # 检查直选
        if rec_numbers == prize_numbers:
            prize_level = "直选"
            prize_amount = PRIZE_VALUES[prize_level]
            total_prize += prize_amount
            prize_counts[prize_level] = prize_counts.get(prize_level, 0) + 1
            winning_details.append({
                'ticket_id': i + 1,
                'numbers': rec_numbers,
                'prize_level': prize_level,
                'amount': prize_amount
            })
            continue
        
        # 检查组选
        rec_set = set(rec_numbers)
        prize_set = set(prize_numbers)
        
        if rec_set == prize_set:
            # 判断组选类型
            if len(rec_set) == 3:
                prize_level = "组选6"  # 三个数字都不同
            else:
                prize_level = "组选3"  # 有重复数字
            
            prize_amount = PRIZE_VALUES[prize_level]
            total_prize += prize_amount
            prize_counts[prize_level] = prize_counts.get(prize_level, 0) + 1
            winning_details.append({
                'ticket_id': i + 1,
                'numbers': rec_numbers,
                'prize_level': prize_level,
                'amount': prize_amount
            })
    
    return total_prize, prize_counts, winning_details

def format_winning_details(winning_details: List[Dict], prize_numbers: List[int], duplex_data: Dict = None, target_period: str = "") -> List[str]:
    """格式化中奖详情为报告字符串，包含复式信息"""
    lines = []
    
    # 添加期号和开奖号码信息
    if target_period:
        lines.append(f"第{target_period}期开奖号码: {prize_numbers[0]}{prize_numbers[1]}{prize_numbers[2]}")
    else:
        lines.append(f"开奖号码: {prize_numbers[0]}{prize_numbers[1]}{prize_numbers[2]}")
    lines.append("")
    
    # 添加中奖详情
    if not winning_details:
        lines.append("本期推荐号码未中奖。")
    else:
        lines.append("🎉 中奖详情:")
        for detail in winning_details:
            numbers_str = f"{detail['numbers'][0]}{detail['numbers'][1]}{detail['numbers'][2]}"
            lines.append(f"第{detail['ticket_id']}注: {numbers_str} - {detail['prize_level']} - {detail['amount']}元")
    
    lines.append("")
    
    # 添加复式推荐信息
    if duplex_data:
        lines.append("📋 复式推荐参考:")
        for position, numbers in duplex_data.items():
            numbers_str = ' '.join(map(str, numbers))
            lines.append(f"  {position}: {numbers_str}")
        lines.append("")
        
        # 计算复式总注数
        if len(duplex_data) == 3:
            total_combinations = len(duplex_data['百位']) * len(duplex_data['十位']) * len(duplex_data['个位'])
            lines.append(f"复式总注数: {total_combinations}注")
    
    return lines

def manage_report(new_entry: Optional[Dict] = None, new_error: Optional[str] = None):
    """管理主报告文件，添加新记录并保持文件大小"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, MAIN_REPORT_FILE)
    
    # 读取现有内容
    existing_content = ""
    if os.path.exists(report_path):
        existing_content = robust_file_read(report_path) or ""
    
    # 准备新内容
    new_content_lines = []
    
    if new_entry:
        new_content_lines.extend([
            f"评估时间: {new_entry['timestamp']}",
            f"评估期号: {new_entry['period']}",
            f"开奖号码: {new_entry['prize_numbers']}",
            f"推荐数量: {new_entry['total_recommendations']}",
            f"中奖注数: {new_entry['winning_count']}",
            f"总奖金: {new_entry['total_prize']}元",
            ""
        ])
        
        if new_entry.get('winning_details'):
            new_content_lines.extend(new_entry['winning_details'])
        
        new_content_lines.extend(["", "=" * 60, ""])
    
    if new_error:
        new_content_lines.extend([
            f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"错误信息: {new_error}",
            "", "=" * 60, ""
        ])
    
    # 合并内容
    if new_content_lines:
        final_content = "\n".join(new_content_lines) + "\n" + existing_content
    else:
        final_content = existing_content
    
    # 写入文件
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        log_message(f"报告已更新: {report_path}")
    except Exception as e:
        log_message(f"写入报告文件失败: {e}", "ERROR")

def main_process():
    """主处理流程"""
    try:
        log_message("开始排列三推荐结果验证...")
        
        # 读取CSV数据
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, CSV_FILE)
        csv_content = robust_file_read(csv_path)
        
        if not csv_content:
            raise Exception(f"无法读取数据文件: {CSV_FILE}")
        
        # 解析数据
        period_data, periods = get_period_data_from_csv(csv_content)
        if not period_data or not periods:
            raise Exception("未能解析到有效的开奖数据")
        
        # 确定评估期和数据截止期
        if len(periods) < 2:
            raise Exception("数据不足，至少需要2期数据")
        
        eval_period = periods[-1]  # 最新期
        data_cutoff_period = periods[-2]  # 倒数第二期
        
        log_message(f"评估期: {eval_period}, 数据截止期: {data_cutoff_period}")
        
        # 查找对应的分析报告
        report_file = find_matching_report(data_cutoff_period)
        if not report_file:
            raise Exception(f"未找到数据截止期为 {data_cutoff_period} 的分析报告")
        
        # 解析推荐号码
        report_content = robust_file_read(report_file)
        if not report_content:
            raise Exception(f"无法读取报告文件: {report_file}")
        
        parsed_data = parse_recommendations_from_report(report_content)
        recommendations = parsed_data['single']
        duplex_data = parsed_data['duplex']
        target_period = parsed_data['target_period']
        
        if not recommendations:
            raise Exception("报告中未找到有效的推荐号码")
        
        # 获取开奖号码
        prize_numbers = period_data[eval_period]['numbers']
        log_message(f"第{eval_period}期开奖号码: {prize_numbers}")
        
        # 计算中奖情况
        total_prize, prize_counts, winning_details = calculate_prize(recommendations, prize_numbers)
        
        # 格式化结果（包含复式信息）
        winning_details_formatted = format_winning_details(
            winning_details, 
            prize_numbers, 
            duplex_data, 
            target_period or eval_period
        )
        
        # 准备报告条目
        report_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'period': eval_period,
            'prize_numbers': f"{prize_numbers[0]}{prize_numbers[1]}{prize_numbers[2]}",
            'total_recommendations': len(recommendations),
            'winning_count': len(winning_details),
            'total_prize': total_prize,
            'winning_details': winning_details_formatted,
            'duplex_info': duplex_data
        }
        
        # 更新主报告
        manage_report(new_entry=report_entry)
        
        # 输出结果
        log_message(f"验证完成！推荐{len(recommendations)}注，中奖{len(winning_details)}注，总奖金{total_prize}元")
        
        if duplex_data:
            duplex_summary = ', '.join([f"{pos}:{len(nums)}个" for pos, nums in duplex_data.items()])
            log_message(f"复式推荐: {duplex_summary}")
        
        if winning_details:
            log_message("中奖详情:")
            for line in winning_details_formatted:
                log_message(f"  {line}")
        
    except Exception as e:
        error_msg = f"验证过程发生错误: {str(e)}"
        log_message(error_msg, "ERROR")
        log_message(traceback.format_exc(), "DEBUG")
        manage_report(new_error=error_msg)

if __name__ == "__main__":
    main_process()