# -*- coding: utf-8 -*-
"""
排列三彩票数据分析与推荐系统
================================

本脚本整合了统计分析、机器学习和策略化组合生成，为排列三彩票提供数据驱动的
号码推荐。脚本支持两种运行模式，由全局变量 `ENABLE_OPTUNA_OPTIMIZATION` 控制：

1.  **分析模式 (默认 `False`)**:
    使用内置的 `DEFAULT_WEIGHTS` 权重，执行一次完整的历史数据分析、策略回测，
    并为下一期生成推荐号码。所有结果会输出到一个带时间戳的详细报告文件中。

2.  **优化模式 (`True`)**:
    在分析前，首先运行 Optuna 框架进行参数搜索，以找到在近期历史数据上
    表现最佳的一组权重。然后，自动使用这组优化后的权重来完成后续的分析、
    回测和推荐。优化过程和结果也会记录在报告中。

版本: 6.0 (PL3 Adapted)
"""

# --- 标准库导入 ---
import os
import sys
import json
import time
import datetime
import logging
import io
import random
from collections import Counter
from contextlib import redirect_stdout
from typing import (Union, Optional, List, Dict, Tuple, Any)
from functools import partial

# --- 第三方库导入 ---
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import concurrent.futures
import warnings

# 抑制sklearn和其他库的警告信息
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ==============================================================================
# --- 全局常量与配置 ---
# ==============================================================================

# --------------------------
# --- 路径与模式配置 ---
# --------------------------
# 脚本文件所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始排列三数据CSV文件路径 (由 pls_data_processor.py 生成)
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'pls.csv')
# 预处理后的数据缓存文件路径，避免每次都重新计算特征
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'pls_processed.csv')

# 运行模式配置:
# True  -> 运行参数优化，耗时较长，但可能找到更优策略。
# False -> 使用默认权重进行快速分析和推荐。
ENABLE_OPTUNA_OPTIMIZATION = True

# --------------------------
# --- 策略开关配置 ---
# --------------------------
# 是否启用最终推荐组合层面的"反向思维"策略 (移除得分最高的几注)
ENABLE_FINAL_COMBO_REVERSE = True
# 在启用反向思维并移除组合后，是否从候选池中补充新的组合以达到目标数量
ENABLE_REVERSE_REFILL = True

# --------------------------
# --- 彩票规则配置 ---
# --------------------------
# 排列三号码的有效范围 (0到9)
NUMBER_RANGE = range(0, 10)
# 排列三位数定义，用于特征工程和模式分析
POSITION_NAMES = ['红球1', '红球2', '红球3']  # 保持与双色球命名一致以复用代码

# --------------------------
# --- 分析与执行参数配置 ---
# --------------------------
# 机器学习模型使用的滞后特征阶数 (e.g., 使用前1、3、5、10期的数据作为特征)
ML_LAG_FEATURES = [1, 3, 5, 8,10]
# 用于生成乘积交互特征的特征对
ML_INTERACTION_PAIRS = [('sum_all', 'odd_count')]
# 用于生成自身平方交互特征的特征
ML_INTERACTION_SELF = ['span']
# 计算号码"近期"出现频率时所参考的期数窗口大小
RECENT_FREQ_WINDOW = 66
# 在分析模式下，进行策略回测时所评估的总期数
BACKTEST_PERIODS_COUNT = 100
# 在优化模式下，每次试验用于快速评估性能的回测期数
OPTIMIZATION_BACKTEST_PERIODS = 20
# 在优化模式下，Optuna 进行参数搜索的总试验次数
OPTIMIZATION_TRIALS = 10
# 训练机器学习模型时，一个号码在历史数据中至少需要出现的次数
MIN_POSITIVE_SAMPLES_FOR_ML = 25

# ==============================================================================
# --- 默认权重配置 (这些参数可被Optuna优化) ---
# ==============================================================================
DEFAULT_WEIGHTS = {
    # --- 反向思维 ---
    'FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT': 0.3,

    # --- 组合生成 ---
    'NUM_COMBINATIONS_TO_GENERATE': 10,
    'TOP_N_NUMBERS_FOR_CANDIDATE': 15,  # 从0-9中选择评分最高的N个

    # --- 号码评分权重 ---
    'FREQ_SCORE_WEIGHT': 25.0,
    'OMISSION_SCORE_WEIGHT': 20.0,
    'MAX_OMISSION_RATIO_SCORE_WEIGHT': 15.0,
    'RECENT_FREQ_SCORE_WEIGHT': 15.0,
    'ML_PROB_SCORE_WEIGHT': 25.0,

    # --- 组合属性匹配奖励 ---
    'COMBINATION_ODD_COUNT_MATCH_BONUS': 10.0,
    'COMBINATION_SIZE_DISTRIBUTION_MATCH_BONUS': 10.0,  # 大中小分布
    'COMBINATION_SUM_RANGE_MATCH_BONUS': 8.0,

    # --- 关联规则挖掘(ARM)参数与奖励 ---
    'ARM_MIN_SUPPORT': 0.005,
    'ARM_MIN_CONFIDENCE': 0.40,
    'ARM_MIN_LIFT': 1.10,
    'ARM_COMBINATION_BONUS_WEIGHT': 15.0,
    'ARM_BONUS_LIFT_FACTOR': 0.50,
    'ARM_BONUS_CONF_FACTOR': 0.30,

    # --- 组合多样性控制 ---
    'DIVERSITY_MIN_DIFFERENT_NUMBERS': 2,  # 改为2，更适合排列三

    # --- 形态多样性控制 ---
    'ENABLE_FORM_TYPE_DIVERSITY': True,    # 新增：启用形态多样性控制
    'TARGET_GROUP3_RATIO': 0.3,            # 新增：目标组三占比
    'FORM_TYPE_MATCH_BONUS': 15.0,         # 新增：形态匹配奖励
}

# ==============================================================================
# --- 机器学习模型参数配置 ---
# ==============================================================================
LGBM_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 80,
    'num_leaves': 12,
    'min_child_samples': 10,
    'lambda_l1': 0.10,
    'lambda_l2': 0.10,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'n_jobs': 1,
    'verbose': -1,
}

# ==============================================================================
# --- 日志系统配置 ---
# ==============================================================================
console_formatter = logging.Formatter('%(message)s')
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# 主日志记录器
logger = logging.getLogger('pls_analyzer')
logger.setLevel(logging.DEBUG)
logger.propagate = False

# 进度日志记录器
progress_logger = logging.getLogger('pls_progress')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False


def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    """设置控制台日志输出级别和格式"""
    console_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            console_handler = handler
            break
    
    if console_handler:
        console_handler.setLevel(level)
        if use_simple_formatter:
            console_handler.setFormatter(console_formatter)
        else:
            console_handler.setFormatter(detailed_formatter)


class SuppressOutput:
    """上下文管理器，用于临时抑制标准输出"""
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.capture_stderr = capture_stderr

    def __enter__(self):
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout:
            sys.stdout.close()
            sys.stdout = self.old_stdout


def get_prize_level(red_1_hit: bool, red_2_hit: bool, red_3_hit: bool, is_direct: bool = True) -> Optional[str]:
    """
    根据排列三中奖情况确定奖级
    
    Args:
        red_1_hit: 百位是否命中
        red_2_hit: 十位是否命中
        red_3_hit: 个位是否命中
        is_direct: 是否直选投注
    
    Returns:
        奖级字符串或None
    """
    if is_direct:
        # 直选：三个位置都要对
        if red_1_hit and red_2_hit and red_3_hit:
            return "直选"
        return None
    else:
        # 组选：只要三个数字都对即可（不考虑顺序）
        hit_count = sum([red_1_hit, red_2_hit, red_3_hit])
        if hit_count == 3:
            return "组选"
        return None


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    加载排列三数据CSV文件
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        DataFrame或None
    """
    if not os.path.exists(file_path):
        logger.error(f"数据文件不存在: {file_path}")
        return None
    
    try:
        # 尝试多种编码读取文件
        encodings = ['utf-8', 'gbk', 'latin-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            logger.error("无法使用任何编码读取数据文件")
            return None
        return df
        
    except Exception as e:
        logger.error(f"加载数据时发生错误: {e}")
        return None


def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    清理和结构化排列三数据
    
    Args:
        df: 原始DataFrame
        
    Returns:
        清理后的DataFrame或None
    """
    if df is None or df.empty:
        logger.error("输入数据为空")
        return None
    
    try:
        # 检查必要的列
        required_columns = ['Seq', 'red_1', 'red_2', 'red_3']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"缺少必要的列: {missing_columns}")
            return None
        
        # 清理数据
        df = df.copy()
        
        # 转换数据类型
        df['Seq'] = pd.to_numeric(df['Seq'], errors='coerce')
        df['red_1'] = pd.to_numeric(df['red_1'], errors='coerce')
        df['red_2'] = pd.to_numeric(df['red_2'], errors='coerce')
        df['red_3'] = pd.to_numeric(df['red_3'], errors='coerce')
        
        # 删除无效行
        initial_count = len(df)
        df = df.dropna()
        
        # 验证数据范围
        valid_mask = (
            (df['red_1'].between(0, 9)) &
            (df['red_2'].between(0, 9)) &
            (df['red_3'].between(0, 9)) &
            (df['Seq'] > 0)
        )
        df = df[valid_mask]
        
        # 确保期号唯一性
        df = df.drop_duplicates(subset=['Seq'], keep='last')
        
        # 按期号排序
        df = df.sort_values('Seq').reset_index(drop=True)
        
        final_count = len(df)
        if final_count == 0:
            logger.error("清理后没有有效数据")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"数据清理过程中发生错误: {e}")
        return None


def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    为排列三数据创建特征工程
    
    Args:
        df: 清理后的DataFrame
        
    Returns:
        带有特征的DataFrame或None
    """
    if df is None or df.empty:
        logger.error("输入数据为空")
        return None
    
    try:
        df = df.copy()
        
        # 基础特征
        df['sum_all'] = df['red_1'] + df['red_2'] + df['red_3']  # 三位数和值
        df['span'] = df[['red_1', 'red_2', 'red_3']].max(axis=1) - df[['red_1', 'red_2', 'red_3']].min(axis=1)  # 跨度
        
        # 奇偶特征
        df['odd_count'] = (df['red_1'] % 2) + (df['red_2'] % 2) + (df['red_3'] % 2)  # 奇数个数
        df['even_count'] = 3 - df['odd_count']  # 偶数个数
        
        # 大小特征（0-4为小，5-9为大）
        df['big_count'] = (df['red_1'] >= 5).astype(int) + (df['red_2'] >= 5).astype(int) + (df['red_3'] >= 5).astype(int)
        df['small_count'] = 3 - df['big_count']
        
        # 质合特征（2,3,5,7为质数，其余为合数）
        prime_numbers = {2, 3, 5, 7}
        df['prime_count'] = 0
        for col in ['red_1', 'red_2', 'red_3']:
            df['prime_count'] += df[col].isin(prime_numbers).astype(int)
        df['composite_count'] = 3 - df['prime_count']
        
        # 重复数字特征
        df['unique_count'] = df[['red_1', 'red_2', 'red_3']].nunique(axis=1)  # 不重复数字个数
        df['repeat_count'] = 3 - df['unique_count']  # 重复数字个数
        
        # 连号特征
        def count_consecutive(row):
            numbers = sorted([row['red_1'], row['red_2'], row['red_3']])
            consecutive = 0
            for i in range(len(numbers) - 1):
                if numbers[i+1] - numbers[i] == 1:
                    consecutive += 1
            return consecutive
        
        df['consecutive_count'] = df.apply(count_consecutive, axis=1)
        
        # 形态特征（组选类型）
        def get_form_type(row):
            numbers = [row['red_1'], row['red_2'], row['red_3']]
            unique_nums = set(numbers)
            if len(unique_nums) == 3:
                return 'group6'  # 组六（三个数字都不同）
            elif len(unique_nums) == 2:
                return 'group3'  # 组三（有两个相同数字）
            else:
                return 'triple'  # 豹子（三个数字相同）
        
        df['form_type'] = df.apply(get_form_type, axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"特征工程过程中发生错误: {e}")
        return None


def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """
    创建滞后特征用于机器学习
    
    Args:
        df: 带特征的DataFrame
        lags: 滞后期数列表
        
    Returns:
        带滞后特征的DataFrame或None
    """
    if df is None or df.empty:
        logger.error("输入数据为空")
        return None
    
    try:
        df = df.copy()
        
        # 为每个滞后期创建特征
        feature_columns = ['sum_all', 'span', 'odd_count', 'big_count', 'prime_count', 'unique_count', 'consecutive_count']
        
        for lag in lags:
            for col in feature_columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # 删除包含NaN的行（由于滞后产生的）
        initial_count = len(df)
        df = df.dropna()
        final_count = len(df)
        
        return df
        
    except Exception as e:
        logger.error(f"创建滞后特征时发生错误: {e}")
        return None


def analyze_frequency_omission(df: pd.DataFrame) -> dict:
    """
    分析号码频率和遗漏情况
    
    Args:
        df: 历史数据DataFrame
        
    Returns:
        包含频率和遗漏分析结果的字典
    """
    if df is None or df.empty:
        logger.error("输入数据为空")
        return {}
    
    try:
        result = {}
        
        # 分析每个位置的号码频率和遗漏
        for pos_idx, pos_name in enumerate(['red_1', 'red_2', 'red_3'], 1):
            pos_analysis = {}
            
            for num in NUMBER_RANGE:
                # 频率分析
                occurrences = (df[pos_name] == num).sum()
                frequency = occurrences / len(df) if len(df) > 0 else 0
                
                # 遗漏分析
                last_occurrence = -1
                for i, value in enumerate(df[pos_name]):
                    if value == num:
                        last_occurrence = i
                
                current_omission = len(df) - 1 - last_occurrence if last_occurrence != -1 else len(df)
                
                # 计算平均遗漏
                omission_periods = []
                last_idx = -1
                for i, value in enumerate(df[pos_name]):
                    if value == num:
                        if last_idx != -1:
                            omission_periods.append(i - last_idx - 1)
                        last_idx = i
                
                avg_omission = np.mean(omission_periods) if omission_periods else len(df)
                max_omission = max(omission_periods) if omission_periods else len(df)
                
                # 近期频率
                recent_data = df[pos_name].tail(RECENT_FREQ_WINDOW)
                recent_frequency = (recent_data == num).sum() / len(recent_data) if len(recent_data) > 0 else 0
                
                pos_analysis[num] = {
                    'frequency': frequency,
                    'occurrences': occurrences,
                    'current_omission': current_omission,
                    'avg_omission': avg_omission,
                    'max_omission': max_omission,
                    'recent_frequency': recent_frequency
                }
            
            result[pos_name] = pos_analysis
        
        return result
        
    except Exception as e:
        logger.error(f"频率和遗漏分析过程中发生错误: {e}")
        return {}


def analyze_patterns(df: pd.DataFrame) -> dict:
    """
    分析排列三的各种模式
    
    Args:
        df: 历史数据DataFrame（应该包含特征工程后的列）
        
    Returns:
        模式分析结果字典
    """
    if df is None or df.empty:
        logger.error("输入数据为空")
        return {}
    
    try:
        result = {}
        
        # 确保数据包含必要的特征列，如果没有则进行特征工程
        required_cols = ['sum_all', 'span', 'odd_count', 'big_count', 'form_type']
        if not all(col in df.columns for col in required_cols):
            logger.warning("数据缺少特征列，尝试重新进行特征工程...")
            df = feature_engineer(df)
            if df is None:
                logger.error("特征工程失败")
                return {}
        
        # 奇偶比模式
        if 'odd_count' in df.columns:
            odd_patterns = df['odd_count'].value_counts().sort_index()
            result['odd_patterns'] = {
                'distribution': odd_patterns.to_dict(),
                'most_common': odd_patterns.idxmax() if not odd_patterns.empty else 2
            }
        
        # 大小比模式
        if 'big_count' in df.columns:
            big_patterns = df['big_count'].value_counts().sort_index()
            result['big_patterns'] = {
                'distribution': big_patterns.to_dict(),
                'most_common': big_patterns.idxmax() if not big_patterns.empty else 1
            }
        
        # 和值分布
        if 'sum_all' in df.columns:
            sum_distribution = df['sum_all'].value_counts().sort_index()
            result['sum_patterns'] = {
                'distribution': sum_distribution.to_dict(),
                'most_common_range': (10, 17),  # 排列三和值最常见范围
                'avg_sum': df['sum_all'].mean()
            }
        
        # 跨度分布
        if 'span' in df.columns:
            span_distribution = df['span'].value_counts().sort_index()
            result['span_patterns'] = {
                'distribution': span_distribution.to_dict(),
                'avg_span': df['span'].mean()
            }
        
        # 新增：形态分布分析
        if 'form_type' in df.columns:
            form_distribution = df['form_type'].value_counts()
            total_count = len(df)
            result['form_patterns'] = {
                'distribution': form_distribution.to_dict(),
                'group3_ratio': form_distribution.get('group3', 0) / total_count if total_count > 0 else 0,
                'group6_ratio': form_distribution.get('group6', 0) / total_count if total_count > 0 else 0,
                'triple_ratio': form_distribution.get('triple', 0) / total_count if total_count > 0 else 0,
                'most_common_form': form_distribution.idxmax() if not form_distribution.empty else 'group6'
            }
        
        return result
        
    except Exception as e:
        logger.error(f"模式分析过程中发生错误: {e}")
        return {}


def analyze_associations(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    """
    关联规则挖掘 - 分析号码之间的关联关系（改进版）
    
    Args:
        df: 历史数据DataFrame
        weights_config: 权重配置字典
        
    Returns:
        关联规则DataFrame
    """
    if df is None or df.empty:
        logger.error("输入数据为空")
        return pd.DataFrame()
    
    try:
        # 确保数据包含form_type特征
        if 'form_type' not in df.columns:
            df = feature_engineer(df)
            if df is None:
                logger.error("特征工程失败，无法进行关联规则分析")
                return pd.DataFrame()
        
        # 改进的事务数据构建方法
        transactions = []
        for _, row in df.iterrows():
            numbers = sorted([int(row['red_1']), int(row['red_2']), int(row['red_3'])])
            transaction = []
            
            # 方法1：添加单个数字（不区分位置）
            for num in numbers:
                transaction.append(f"has_{num}")
            
            # 方法2：添加数字对关系（更强的关联性）
            unique_numbers = list(set(numbers))
            for i in range(len(unique_numbers)):
                for j in range(i+1, len(unique_numbers)):
                    transaction.append(f"pair_{unique_numbers[i]}_{unique_numbers[j]}")
            
            # 方法3：添加形态特征
            form_type = row.get('form_type', 'unknown')
            if form_type != 'unknown':
                transaction.append(f"form_{form_type}")
            
            # 方法4：添加和值范围特征
            sum_val = sum(numbers)
            if sum_val <= 10:
                transaction.append("sum_low")
            elif sum_val <= 17:
                transaction.append("sum_mid")
            else:
                transaction.append("sum_high")
            
            # 方法5：添加奇偶特征
            odd_count = sum(1 for num in numbers if num % 2 == 1)
            transaction.append(f"odd_count_{odd_count}")
            
            # 方法6：添加大小特征
            big_count = sum(1 for num in numbers if num >= 5)
            transaction.append(f"big_count_{big_count}")
            
            # 方法7：添加跨度特征
            span = max(numbers) - min(numbers)
            if span <= 3:
                transaction.append("span_small")
            elif span <= 6:
                transaction.append("span_mid")
            else:
                transaction.append("span_large")
            
            transactions.append(transaction)
        
        if not transactions:
            logger.warning("没有有效的交易数据用于关联规则挖掘")
            return pd.DataFrame()
        
        # 使用TransactionEncoder转换数据
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # 应用Apriori算法 - 使用更宽松但合理的参数
        min_support = max(0.002, weights_config.get('ARM_MIN_SUPPORT', 0.01))  # 最低0.2%支持度
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            logger.warning(f"未发现满足最小支持度({min_support})的频繁项集")
            return pd.DataFrame()
        
        # 生成关联规则 - 使用更现实的置信度阈值
        min_confidence = max(0.1, weights_config.get('ARM_MIN_CONFIDENCE', 0.40))  # 最低10%置信度
        try:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        except ValueError as e:
            logger.warning(f"关联规则生成失败: {e}")
            return pd.DataFrame()
        
        if rules.empty:
            logger.warning(f"未发现满足最小置信度({min_confidence})的关联规则")
            return pd.DataFrame()
        
        # 过滤提升度 - 使用更现实的提升度阈值
        min_lift = max(1.0, weights_config.get('ARM_MIN_LIFT', 1.20))  # 最低1.0提升度（无负面效果）
        rules = rules[rules['lift'] >= min_lift]
        
        if rules.empty:
            logger.warning(f"未发现满足最小提升度({min_lift})的关联规则")
            return pd.DataFrame()
        
        # 按提升度排序，保留最有价值的规则
        rules = rules.sort_values('lift', ascending=False)
        
        # 限制规则数量以提高性能
        max_rules = 1000
        if len(rules) > max_rules:
            rules = rules.head(max_rules)
        
        logger.info(f"发现 {len(rules)} 条有效关联规则 (支持度≥{min_support}, 置信度≥{min_confidence}, 提升度≥{min_lift})")
        
        return rules
        
    except Exception as e:
        logger.error(f"关联规则分析过程中发生错误: {e}")
        return pd.DataFrame()


def calculate_scores(freq_data: Dict, probabilities: Dict, weights: Dict) -> Dict[str, Dict[int, float]]:
    """
    计算每个位置每个号码的综合得分
    
    Args:
        freq_data: 频率分析数据
        probabilities: 机器学习预测概率
        weights: 权重配置
        
    Returns:
        每个位置每个号码的得分字典
    """
    try:
        scores = {}
        
        for pos_name in ['red_1', 'red_2', 'red_3']:
            position_scores = {}
            pos_freq_data = freq_data.get(pos_name, {})
            pos_ml_probs = probabilities.get(pos_name, {})
            
            for num in NUMBER_RANGE:
                num_data = pos_freq_data.get(num, {})
                ml_prob = pos_ml_probs.get(num, 0.1)  # 默认概率
                
                # 各项得分计算
                freq_score = num_data.get('frequency', 0) * 100
                
                # 遗漏得分（当前遗漏越接近平均遗漏得分越高）
                current_omission = num_data.get('current_omission', 0)
                avg_omission = num_data.get('avg_omission', 1)
                omission_ratio = current_omission / max(avg_omission, 1)
                omission_score = max(0, 100 - abs(omission_ratio - 1) * 50)
                
                # 最大遗漏比率得分
                max_omission = num_data.get('max_omission', 1)
                max_omission_ratio = current_omission / max(max_omission, 1)
                max_omission_score = max_omission_ratio * 100
                
                # 近期频率得分
                recent_freq_score = num_data.get('recent_frequency', 0) * 100
                
                # 机器学习概率得分
                ml_score = ml_prob * 100
                
                # 综合得分计算
                total_score = (
                    freq_score * weights.get('FREQ_SCORE_WEIGHT', 25.0) +
                    omission_score * weights.get('OMISSION_SCORE_WEIGHT', 20.0) +
                    max_omission_score * weights.get('MAX_OMISSION_RATIO_SCORE_WEIGHT', 15.0) +
                    recent_freq_score * weights.get('RECENT_FREQ_SCORE_WEIGHT', 15.0) +
                    ml_score * weights.get('ML_PROB_SCORE_WEIGHT', 25.0)
                )
                
                position_scores[num] = total_score
            
            scores[pos_name] = position_scores
        
        # 归一化得分
        def normalize_scores(scores_dict):
            for pos_name in scores_dict:
                pos_scores = scores_dict[pos_name]
                if pos_scores:
                    max_score = max(pos_scores.values())
                    min_score = min(pos_scores.values())
                    score_range = max_score - min_score
                    if score_range > 0:
                        for num in pos_scores:
                            pos_scores[num] = ((pos_scores[num] - min_score) / score_range) * 100
            return scores_dict
        
        scores = normalize_scores(scores)
        return scores
        
    except Exception as e:
        logger.error(f"计算评分时发生错误: {e}")
        return {}


def train_single_lgbm_model(position_name: str, number: int, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Optional[LGBMClassifier], Optional[str]]:
    """
    训练单个位置单个号码的LightGBM模型
    
    Args:
        position_name: 位置名称
        number: 号码
        X_train: 训练特征
        y_train: 训练标签
        
    Returns:
        训练好的模型和错误信息
    """
    try:
        positive_samples = y_train.sum()
        if positive_samples < MIN_POSITIVE_SAMPLES_FOR_ML:
            return None, f"正样本数量不足 ({positive_samples} < {MIN_POSITIVE_SAMPLES_FOR_ML})"
        
        with SuppressOutput():
            model = LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_train, y_train)
        
        return model, None
        
    except Exception as e:
        return None, str(e)


def train_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int]) -> Optional[Dict[str, Any]]:
    """
    训练排列三预测模型
    
    Args:
        df_train_raw: 原始训练数据
        ml_lags_list: 滞后特征列表
        
    Returns:
        训练好的模型字典或None
    """
    try:
        # 准备训练数据
        df_with_features = feature_engineer(df_train_raw)
        if df_with_features is None:
            return None
        
        df_with_lags = create_lagged_features(df_with_features, ml_lags_list)
        if df_with_lags is None:
            return None
        
        # 准备特征矩阵
        feature_columns = [col for col in df_with_lags.columns 
                          if col not in ['Seq', 'red_1', 'red_2', 'red_3', 'form_type']]
        X = df_with_lags[feature_columns]
        
        models = {}
        
        # 为每个位置的每个号码训练模型
        for pos_name in ['red_1', 'red_2', 'red_3']:
            position_models = {}
            
            for num in NUMBER_RANGE:
                # 创建标签（该号码是否出现在该位置）
                y = (df_with_lags[pos_name] == num).astype(int)
                
                # 训练模型
                model, error = train_single_lgbm_model(pos_name, num, X, y)
                if model is not None:
                    position_models[num] = model
            
            models[pos_name] = position_models
        
        return models
        
    except Exception as e:
        logger.error(f"训练预测模型时发生错误: {e}")
        return None


def predict_next_draw_probabilities(df_historical: pd.DataFrame, trained_models: Optional[Dict], ml_lags_list: List[int]) -> Dict[str, Dict[int, float]]:
    """
    预测下一期各位置各号码的出现概率
    
    Args:
        df_historical: 历史数据
        trained_models: 训练好的模型
        ml_lags_list: 滞后特征列表
        
    Returns:
        预测概率字典
    """
    try:
        if trained_models is None:
            logger.warning("没有可用的训练模型，使用默认概率")
            return {pos_name: {num: 0.1 for num in NUMBER_RANGE} for pos_name in ['red_1', 'red_2', 'red_3']}
        
        # 准备预测数据
        df_with_features = feature_engineer(df_historical)
        if df_with_features is None:
            return {}
        
        df_with_lags = create_lagged_features(df_with_features, ml_lags_list)
        if df_with_lags is None or df_with_lags.empty:
            return {}
        
        # 获取最新一行作为预测特征（保持DataFrame格式以确保特征名称一致）
        feature_columns = [col for col in df_with_lags.columns 
                          if col not in ['Seq', 'red_1', 'red_2', 'red_3', 'form_type']]
        latest_features = df_with_lags[feature_columns].iloc[-1:]
        
        predictions = {}
        
        for pos_name in ['red_1', 'red_2', 'red_3']:
            position_predictions = {}
            position_models = trained_models.get(pos_name, {})
            
            for num in NUMBER_RANGE:
                model = position_models.get(num)
                if model is not None:
                    try:
                        with SuppressOutput():
                            prob = model.predict_proba(latest_features)[0][1]  # 正类概率
                        position_predictions[num] = prob
                    except Exception:
                        position_predictions[num] = 0.1  # 默认概率
                else:
                    position_predictions[num] = 0.1  # 默认概率
            
            predictions[pos_name] = position_predictions
        
        return predictions
        
    except Exception as e:
        logger.error(f"预测概率时发生错误: {e}")
        return {}


def generate_combinations(scores_data: Dict, pattern_data: Dict, arm_rules: pd.DataFrame, weights_config: Dict) -> Tuple[List[Dict], List[str]]:
    """
    生成排列三推荐组合
    
    Args:
        scores_data: 号码得分数据
        pattern_data: 模式分析数据
        arm_rules: 关联规则
        weights_config: 权重配置
        
    Returns:
        推荐组合列表和详细信息
    """
    try:
        # 获取各位置的候选号码
        candidates = {}
        top_n = weights_config.get('TOP_N_NUMBERS_FOR_CANDIDATE', 15)
        
        for pos_name in ['red_1', 'red_2', 'red_3']:
            position_scores = scores_data.get(pos_name, {})
            if position_scores:
                # 按得分排序选择候选号码
                sorted_numbers = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)
                candidates[pos_name] = [num for num, score in sorted_numbers[:min(top_n, len(sorted_numbers))]]
            else:
                candidates[pos_name] = list(NUMBER_RANGE)[:top_n]
        
        # 生成所有可能的组合
        all_combinations = []
        for pos1 in candidates['red_1']:
            for pos2 in candidates['red_2']:
                for pos3 in candidates['red_3']:
                    combination = [pos1, pos2, pos3]
                    
                    # 计算组合得分
                    combo_score = (
                        scores_data.get('red_1', {}).get(pos1, 0) +
                        scores_data.get('red_2', {}).get(pos2, 0) +
                        scores_data.get('red_3', {}).get(pos3, 0)
                    )
                    
                    # 模式匹配奖励
                    pattern_bonus = calculate_pattern_bonus(combination, pattern_data, weights_config)
                    
                    # 关联规则奖励
                    arm_bonus = calculate_arm_bonus(combination, arm_rules, weights_config)
                    
                    total_score = combo_score + pattern_bonus + arm_bonus
                    
                    all_combinations.append({
                        'numbers': combination,
                        'score': total_score,
                        'base_score': combo_score,
                        'pattern_bonus': pattern_bonus,
                        'arm_bonus': arm_bonus
                    })
        
        # 按得分排序
        all_combinations.sort(key=lambda x: x['score'], reverse=True)
        
        # 应用多样性控制
        final_combinations = apply_diversity_control(all_combinations, weights_config)
        
        # 应用反向思维（如果启用）
        if ENABLE_FINAL_COMBO_REVERSE:
            final_combinations = apply_reverse_thinking(final_combinations, weights_config)
        
        # 限制输出数量
        target_count = weights_config.get('NUM_COMBINATIONS_TO_GENERATE', 10)
        final_combinations = final_combinations[:target_count]
        
        # 生成详细信息
        details = []
        for i, combo in enumerate(final_combinations):
            details.append(
                f"注 {i+1}: {combo['numbers']} - "
                f"总分{combo['score']:.1f} (基础{combo['base_score']:.1f} + "
                f"模式{combo['pattern_bonus']:.1f} + ARM{combo['arm_bonus']:.1f})"
            )
        
        return final_combinations, details
        
    except Exception as e:
        logger.error(f"生成组合时发生错误: {e}")
        return [], []


def calculate_pattern_bonus(combination: List[int], pattern_data: Dict, weights_config: Dict) -> float:
    """计算模式匹配奖励"""
    try:
        bonus = 0.0
        
        # 奇偶比匹配
        odd_count = sum(num % 2 for num in combination)
        most_common_odd = pattern_data.get('odd_patterns', {}).get('most_common', 2)
        if odd_count == most_common_odd:
            bonus += weights_config.get('COMBINATION_ODD_COUNT_MATCH_BONUS', 10.0)
        
        # 大小比匹配
        big_count = sum(1 for num in combination if num >= 5)
        most_common_big = pattern_data.get('big_patterns', {}).get('most_common', 1)
        if big_count == most_common_big:
            bonus += weights_config.get('COMBINATION_SIZE_DISTRIBUTION_MATCH_BONUS', 10.0)
        
        # 和值范围匹配
        sum_value = sum(combination)
        common_range = pattern_data.get('sum_patterns', {}).get('most_common_range', (10, 17))
        if common_range[0] <= sum_value <= common_range[1]:
            bonus += weights_config.get('COMBINATION_SUM_RANGE_MATCH_BONUS', 8.0)
        
        # 新增：形态匹配奖励
        if weights_config.get('ENABLE_FORM_TYPE_DIVERSITY', False):
            unique_count = len(set(combination))
            form_patterns = pattern_data.get('form_patterns', {})
            
            # 根据历史分布给予形态奖励
            if unique_count == 2:  # 组三形式
                group3_ratio = form_patterns.get('group3_ratio', 0.3)
                # 如果组三在历史中出现频率较高，给予额外奖励
                if group3_ratio > 0.2:  # 如果组三历史占比超过20%
                    bonus += weights_config.get('FORM_TYPE_MATCH_BONUS', 15.0) * group3_ratio
            elif unique_count == 3:  # 组六形式
                group6_ratio = form_patterns.get('group6_ratio', 0.6)
                # 给予组六相应的奖励
                bonus += weights_config.get('FORM_TYPE_MATCH_BONUS', 15.0) * group6_ratio * 0.5  # 组六奖励降低
        
        return bonus
        
    except Exception:
        return 0.0


def calculate_arm_bonus(combination: List[int], arm_rules: pd.DataFrame, weights_config: Dict) -> float:
    """计算关联规则匹配奖励（改进版）"""
    try:
        if arm_rules.empty:
            return 0.0
        
        bonus = 0.0
        
        # 构建当前组合的特征集合
        numbers = sorted(combination)
        current_features = set()
        
        # 1. 单个数字特征
        for num in numbers:
            current_features.add(f"has_{num}")
        
        # 2. 数字对特征
        unique_numbers = list(set(numbers))
        for i in range(len(unique_numbers)):
            for j in range(i+1, len(unique_numbers)):
                current_features.add(f"pair_{unique_numbers[i]}_{unique_numbers[j]}")
        
        # 3. 形态特征
        unique_count = len(set(numbers))
        if unique_count == 2:
            current_features.add("form_group3")
        elif unique_count == 3:
            current_features.add("form_group6")
        else:
            current_features.add("form_triple")
        
        # 4. 和值范围特征
        sum_val = sum(numbers)
        if sum_val <= 10:
            current_features.add("sum_low")
        elif sum_val <= 17:
            current_features.add("sum_mid")
        else:
            current_features.add("sum_high")
        
        # 5. 奇偶特征
        odd_count = sum(1 for num in numbers if num % 2 == 1)
        current_features.add(f"odd_count_{odd_count}")
        
        # 6. 大小特征
        big_count = sum(1 for num in numbers if num >= 5)
        current_features.add(f"big_count_{big_count}")
        
        # 7. 跨度特征
        span = max(numbers) - min(numbers)
        if span <= 3:
            current_features.add("span_small")
        elif span <= 6:
            current_features.add("span_mid")
        else:
            current_features.add("span_large")
        
        # 计算关联规则匹配奖励
        base_weight = weights_config.get('ARM_COMBINATION_BONUS_WEIGHT', 15.0)
        lift_factor = weights_config.get('ARM_BONUS_LIFT_FACTOR', 0.50)
        conf_factor = weights_config.get('ARM_BONUS_CONF_FACTOR', 0.30)
        
        matched_rules_count = 0
        total_rule_strength = 0.0
        
        for _, rule in arm_rules.iterrows():
            try:
                antecedents = set(rule['antecedents'])
                consequents = set(rule['consequents'])
                
                # 检查规则是否匹配当前组合
                if antecedents.issubset(current_features) and consequents.issubset(current_features):
                    matched_rules_count += 1
                    
                    # 计算规则强度
                    rule_strength = (
                        rule['confidence'] * conf_factor +
                        min(rule['lift'], 10.0) * lift_factor  # 限制提升度以避免极端值
                    )
                    total_rule_strength += rule_strength
                    
                    # 根据规则类型给予不同权重
                    rule_bonus = base_weight
                    
                    # 形态相关规则给予更高权重
                    if any('form_' in str(item) for item in antecedents | consequents):
                        rule_bonus *= 1.5
                    
                    # 数字对相关规则给予中等权重
                    elif any('pair_' in str(item) for item in antecedents | consequents):
                        rule_bonus *= 1.2
                    
                    # 单数字相关规则给予基础权重
                    elif any('has_' in str(item) for item in antecedents | consequents):
                        rule_bonus *= 1.0
                    
                    bonus += rule_bonus * (rule_strength / 10.0)  # 归一化处理
                    
            except Exception as e:
                # 跳过无法处理的规则
                continue
        
        # 如果匹配了很多规则，给予额外奖励
        if matched_rules_count >= 5:
            bonus *= 1.3
        elif matched_rules_count >= 3:
            bonus *= 1.1
        
        # 限制最大奖励值，避免过度影响
        max_bonus = base_weight * 5
        bonus = min(bonus, max_bonus)
        
        return bonus
        
    except Exception as e:
        logger.debug(f"ARM奖励计算失败: {e}")
        return 0.0


def apply_diversity_control(combinations: List[Dict], weights_config: Dict) -> List[Dict]:
    """应用多样性控制，确保推荐组合有足够差异和形态多样性"""
    try:
        min_different = weights_config.get('DIVERSITY_MIN_DIFFERENT_NUMBERS', 2)
        enable_form_diversity = weights_config.get('ENABLE_FORM_TYPE_DIVERSITY', True)
        target_group3_ratio = weights_config.get('TARGET_GROUP3_RATIO', 0.3)
        
        selected = []
        form_counts = {'group3': 0, 'group6': 0, 'triple': 0}
        
        # 首先按得分排序
        combinations_sorted = sorted(combinations, key=lambda x: x['score'], reverse=True)
        
        for combo in combinations_sorted:
            is_diverse = True
            current_numbers = set(combo['numbers'])
            
            # 检查数字多样性
            for selected_combo in selected:
                selected_numbers = set(selected_combo['numbers'])
                different_count = len(current_numbers.symmetric_difference(selected_numbers))
                if different_count < min_different:
                    is_diverse = False
                    break
            
            if not is_diverse:
                continue
            
            # 检查形态多样性（如果启用）
            if enable_form_diversity:
                unique_count = len(current_numbers)
                if unique_count == 2:
                    current_form = 'group3'
                elif unique_count == 3:
                    current_form = 'group6'
                else:
                    current_form = 'triple'
                
                # 计算当前形态比例
                total_selected = len(selected)
                if total_selected > 0:
                    current_group3_ratio = form_counts['group3'] / total_selected
                    # 如果组三比例已经足够或者当前不是组三，则正常添加
                    # 如果组三比例不足且当前是组三，则优先添加
                    if current_form == 'group3' and current_group3_ratio < target_group3_ratio:
                        # 优先添加组三号码
                        pass
                    elif current_form != 'group3' and current_group3_ratio >= target_group3_ratio:
                        # 如果组三比例已够，优先添加其他形态
                        pass
                    elif current_form == 'group3' and current_group3_ratio >= target_group3_ratio:
                        # 如果组三比例已够，且当前是组三，则跳过（除非总数不够）
                        if total_selected >= weights_config.get('NUM_COMBINATIONS_TO_GENERATE', 10) * 0.7:
                            continue
                
                form_counts[current_form] += 1
            
            selected.append(combo)
        
        return selected
        
    except Exception:
        return combinations


def apply_reverse_thinking(combinations: List[Dict], weights_config: Dict) -> List[Dict]:
    """应用反向思维策略"""
    try:
        if not combinations:
            return combinations
        
        remove_percent = weights_config.get('FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT', 0.3)
        remove_count = max(1, int(len(combinations) * remove_percent))
        
        # 移除得分最高的几个组合
        remaining = combinations[remove_count:]
        
        # 如果启用补充，从候选池中补充
        if ENABLE_REVERSE_REFILL and len(remaining) < len(combinations):
            # 这里可以实现补充逻辑，暂时直接返回剩余组合
            pass
        
        return remaining
        
    except Exception:
        return combinations


def run_analysis_and_recommendation(df_hist: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame) -> Tuple:
    """
    执行一次完整的分析和推荐流程，用于特定一期。

    Returns:
        tuple: 包含推荐组合、输出字符串、分析摘要、训练模型和分数的元组。
    """
    freq_data = analyze_frequency_omission(df_hist)
    patt_data = analyze_patterns(df_hist)
    ml_models = train_prediction_models(df_hist, ml_lags)
    probabilities = predict_next_draw_probabilities(df_hist, ml_models, ml_lags) if ml_models else {'red_1': {}, 'red_2': {}, 'red_3': {}}
    scores = calculate_scores(freq_data, probabilities, weights_config)
    recs, rec_strings = generate_combinations(scores, patt_data, arm_rules, weights_config)
    analysis_summary = {'frequency_omission': freq_data, 'patterns': patt_data}
    return recs, rec_strings, analysis_summary, ml_models, scores


def run_backtest(full_df: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame, num_periods: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    在历史数据上执行策略回测，以评估策略表现。

    Returns:
        tuple: 包含详细回测结果的DataFrame和统计摘要的字典。
    """
    min_data_needed = (max(ml_lags) if ml_lags else 0) + MIN_POSITIVE_SAMPLES_FOR_ML + num_periods
    if len(full_df) < min_data_needed:
        logger.error(f"数据不足以回测{num_periods}期。需要至少{min_data_needed}期，当前有{len(full_df)}期。")
        return pd.DataFrame(), {}

    start_idx = len(full_df) - num_periods
    results, prize_counts = [], Counter()
    best_hits_per_period = []
    
    logger.info("策略回测已启动...")
    start_time = time.time()
    
    for i in range(num_periods):
        current_iter = i + 1
        current_idx = start_idx + i
        
        # 使用SuppressOutput避免在回测循环中打印大量日志
        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
            hist_data = full_df.iloc[:current_idx]
            predicted_combos, _, _, _, _ = run_analysis_and_recommendation(hist_data, ml_lags, weights_config, arm_rules)
            
        actual_outcome = full_df.loc[current_idx]
        actual_numbers = [actual_outcome['red_1'], actual_outcome['red_2'], actual_outcome['red_3']]
        
        period_max_hits, period_best_combo = 0, None
        if not predicted_combos:
            best_hits_per_period.append({'period': actual_outcome['Seq'], 'best_hits': 0, 'best_combo': None, 'prize': None})
        else:
            for combo_dict in predicted_combos:
                combo = combo_dict['numbers']
                
                # 直选验证（位置完全匹配）
                direct_hits = sum(1 for j in range(3) if combo[j] == actual_numbers[j])
                direct_prize = get_prize_level(combo[0] == actual_numbers[0], 
                                               combo[1] == actual_numbers[1], 
                                               combo[2] == actual_numbers[2], 
                                               is_direct=True)
                
                # 组选验证（数字匹配，不考虑顺序）
                group_hits = len(set(combo) & set(actual_numbers))
                group_prize = None
                if group_hits == 3:
                    # 检查是组三还是组六
                    if len(set(actual_numbers)) == 2:  # 组三
                        group_prize = get_prize_level(True, True, True, is_direct=False)
                        group_prize = "组选3"
                    else:  # 组六
                        group_prize = "组选6"
                
                # 选择最好的中奖方式
                best_prize = direct_prize if direct_prize else group_prize
                if best_prize:
                    prize_counts[best_prize] += 1
                
                # 记录结果
                total_hits = max(direct_hits, group_hits)
                results.append({
                    'period': actual_outcome['Seq'], 
                    'direct_hits': direct_hits,
                    'group_hits': group_hits,
                    'total_hits': total_hits,
                    'prize': best_prize
                })
                
                if total_hits > period_max_hits:
                    period_max_hits = total_hits
                    period_best_combo = combo
            
            best_hits_per_period.append({
                'period': actual_outcome['Seq'], 
                'best_hits': period_max_hits, 
                'best_combo': period_best_combo,
                'prize': get_prize_level(period_best_combo[0] == actual_numbers[0] if period_best_combo else False,
                                        period_best_combo[1] == actual_numbers[1] if period_best_combo else False,
                                        period_best_combo[2] == actual_numbers[2] if period_best_combo else False,
                                        is_direct=True) if period_best_combo else None
            })

        # 简化进度记录
        if current_iter == 1 or current_iter % 10 == 0 or current_iter == num_periods:
            logger.info("策略回测已启动...")
            
    return pd.DataFrame(results), {'prize_counts': dict(prize_counts), 'best_hits_per_period': pd.DataFrame(best_hits_per_period)}


# ==============================================================================
# --- Optuna 参数优化模块 ---
# ==============================================================================

def objective(trial: optuna.trial.Trial, df_for_opt: pd.DataFrame, ml_lags: List[int], arm_rules: pd.DataFrame) -> float:
    """Optuna 的目标函数，用于评估一组给定的权重参数的好坏。"""
    trial_weights = {}
    
    # 动态地从DEFAULT_WEIGHTS构建搜索空间
    for key, value in DEFAULT_WEIGHTS.items():
        if isinstance(value, int):
            if 'NUM_COMBINATIONS' in key: 
                trial_weights[key] = trial.suggest_int(key, 5, 15)
            elif 'TOP_N' in key: 
                trial_weights[key] = trial.suggest_int(key, 10, 20)  # 适配排列三的0-9范围
            elif 'DIVERSITY_MIN_DIFFERENT_NUMBERS' in key:
                trial_weights[key] = trial.suggest_int(key, 1, 3)  # 针对排列三调整范围
            else: 
                trial_weights[key] = trial.suggest_int(key, max(0, value - 2), value + 2)
        elif isinstance(value, float):
            # 对不同类型的浮点数使用不同的搜索范围
            if any(k in key for k in ['PERCENT', 'FACTOR', 'SUPPORT', 'CONFIDENCE']):
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 1.5)
            elif 'TARGET_GROUP3_RATIO' in key:
                trial_weights[key] = trial.suggest_float(key, 0.1, 0.5)  # 组三目标比例
            else: # 对权重参数使用更宽的搜索范围
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 2.0)
        elif isinstance(value, bool):
            trial_weights[key] = trial.suggest_categorical(key, [True, False])

    full_trial_weights = DEFAULT_WEIGHTS.copy()
    full_trial_weights.update(trial_weights)
    
    # 在快速回测中评估这组权重
    with SuppressOutput():
        _, backtest_stats = run_backtest(df_for_opt, ml_lags, full_trial_weights, arm_rules, OPTIMIZATION_BACKTEST_PERIODS)
        
    # 定义一个分数来衡量表现，高奖金等级的权重更高
    # 新增：对组三给予更高权重，鼓励系统生成组三号码
    prize_weights = {'直选': 1040, '组选3': 346 * 1.5, '组选6': 173}  # 组三权重提高50%
    score = sum(prize_weights.get(p, 0) * c for p, c in backtest_stats.get('prize_counts', {}).items())
    return score


def optuna_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial, total_trials: int):
    """Optuna 的回调函数，用于在控制台报告优化进度。"""
    # 简化日志输出，类似双色球的格式
    logger.info("策略回测已启动...")


def main():
    """主程序入口"""
    # 1. 初始化日志记录器，同时输出到控制台和文件
    log_filename = os.path.join(SCRIPT_DIR, f"pls_analysis_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    try:
        file_handler = logging.FileHandler(log_filename, 'w', 'utf-8')
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        set_console_verbosity(logging.INFO, use_simple_formatter=True)
                    
    except Exception as log_init_error:
        print(f"[ERROR] 日志初始化失败: {log_init_error}")
        # 创建一个基本的控制台日志记录器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("--- 排列三数据分析与推荐系统 ---")
    logger.info("启动数据加载和预处理...")

    # 2. 健壮的数据加载逻辑
    main_df = None
    if os.path.exists(PROCESSED_CSV_PATH):
        main_df = load_data(PROCESSED_CSV_PATH)

    if main_df is None or main_df.empty:
        logger.info("未找到或无法加载缓存数据，正在从原始文件生成...")
        raw_df = load_data(CSV_FILE_PATH)
        if raw_df is not None and not raw_df.empty:
            logger.info("原始数据加载成功，开始清洗...")
            cleaned_df = clean_and_structure(raw_df)
            if cleaned_df is not None and not cleaned_df.empty:
                logger.info("数据清洗成功，开始特征工程...")
                main_df = feature_engineer(cleaned_df)
                if main_df is not None and not main_df.empty:
                    logger.info("特征工程成功，保存预处理数据...")
                    try:
                        main_df.to_csv(PROCESSED_CSV_PATH, index=False)
                        logger.info(f"预处理数据已保存到: {PROCESSED_CSV_PATH}")
                    except IOError as e:
                        logger.error(f"保存预处理数据失败: {e}")
    
    if main_df is None or main_df.empty:
        logger.critical("数据准备失败，无法继续。程序终止。")
        sys.exit(1)
    
    logger.info(f"数据加载完成，共 {len(main_df)} 期有效数据。")
    last_period = main_df['Seq'].iloc[-1]

    # 3. 根据模式执行：优化或直接分析
    active_weights = DEFAULT_WEIGHTS.copy()
    optuna_summary = None

    if ENABLE_OPTUNA_OPTIMIZATION:
        logger.info("\n" + "="*25 + " Optuna 参数优化模式 " + "="*25)
        set_console_verbosity(logging.INFO, use_simple_formatter=False)
        
        # 优化前先进行一次全局关联规则分析
        optuna_arm_rules = analyze_associations(main_df, DEFAULT_WEIGHTS)
        
        study = optuna.create_study(direction="maximize")
        global OPTUNA_START_TIME; OPTUNA_START_TIME = time.time()
        progress_callback_with_total = partial(optuna_progress_callback, total_trials=OPTIMIZATION_TRIALS)
        
        try:
            study.optimize(lambda t: objective(t, main_df, ML_LAG_FEATURES, optuna_arm_rules), n_trials=OPTIMIZATION_TRIALS, callbacks=[progress_callback_with_total])
            logger.info("Optuna 优化完成。")
            active_weights.update(study.best_params)
            optuna_summary = {"status": "完成", "best_value": study.best_value, "best_params": study.best_params}
        except Exception as e:
            logger.error(f"Optuna 优化过程中断: {e}", exc_info=True)
            optuna_summary = {"status": "中断", "error": str(e)}
            logger.warning("优化中断，将使用默认权重继续分析。")
    
    # 4. 切换到报告模式并打印报告头
    report_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(report_formatter)
    console_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            console_handler = handler
            break
    if console_handler:
        console_handler.setFormatter(report_formatter)
    
    logger.info("\n\n" + "="*60 + f"\n{' ' * 18}排列三策略分析报告\n" + "="*60)
    logger.info(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"分析基于数据: 截至 {last_period} 期 (共 {len(main_df)} 期)")
    logger.info(f"本次预测目标: 第 {last_period + 1} 期")
    logger.info(f"日志文件: {os.path.basename(log_filename)}")

    # 5. 打印优化摘要
    if ENABLE_OPTUNA_OPTIMIZATION and optuna_summary:
        logger.info("\n" + "="*25 + " Optuna 优化摘要 " + "="*25)
        logger.info(f"优化状态: {optuna_summary['status']}")
        if optuna_summary['status'] == '完成':
            logger.info(f"最佳性能得分: {optuna_summary['best_value']:.4f}")
            logger.info("--- 本次分析已采用以下优化参数 ---")
            import json
            best_params_str = json.dumps(optuna_summary['best_params'], indent=2, ensure_ascii=False)
            logger.info(best_params_str)
        else: logger.info(f"错误信息: {optuna_summary['error']}")
    else:
        logger.info("\n--- 本次分析使用脚本内置的默认权重 ---")

    # 6. 全局分析
    full_history_arm_rules = analyze_associations(main_df, active_weights)
    
    # 7. 回测并打印报告
    logger.info("\n" + "="*25 + " 策 略 回 测 摘 要 " + "="*25)
    backtest_results_df, backtest_stats = run_backtest(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules, BACKTEST_PERIODS_COUNT)
    
    if not backtest_results_df.empty:
        num_periods_tested = len(backtest_results_df['period'].unique())
        num_combos_per_period = active_weights.get('NUM_COMBINATIONS_TO_GENERATE', 10)
        total_bets = len(backtest_results_df)
        logger.info(f"回测周期: 最近 {num_periods_tested} 期 | 每期注数: {num_combos_per_period} | 总投入注数: {total_bets}")
        logger.info("\n--- 1. 奖金与回报分析 ---")
        prize_dist, prize_values = backtest_stats.get('prize_counts', {}), {'直选': 1040, '组选3': 346, '组选6': 173}
        total_revenue = sum(prize_values.get(p, 0) * c for p, c in prize_dist.items())
        total_cost = total_bets * 2
        roi = (total_revenue - total_cost) * 100 / total_cost if total_cost > 0 else 0
        logger.info(f"  - 估算总回报: {total_revenue:,.2f} 元 (总成本: {total_cost:,.2f} 元)")
        logger.info(f"  - 投资回报率 (ROI): {roi:.2f}%")
        logger.info("  - 中奖等级分布 (总计):")
        if prize_dist:
            for prize in prize_values.keys():
                if prize in prize_dist: logger.info(f"    - {prize:<4s}: {prize_dist[prize]:>4d} 次")
        else: logger.info("    - 未命中任何奖级。")
        logger.info("\n--- 2. 核心性能指标 ---")
        logger.info(f"  - 平均直选命中 (每注): {backtest_results_df['direct_hits'].mean():.3f} / 3")
        logger.info(f"  - 平均组选命中 (每注): {backtest_results_df['group_hits'].mean():.3f} / 3")
        logger.info("\n--- 3. 每期最佳命中表现 ---")
        if (best_hits_df := backtest_stats.get('best_hits_per_period')) is not None and not best_hits_df.empty:
            logger.info("  - 在一期内至少命中:")
            direct_hits = best_hits_df[best_hits_df['prize'] == '直选'].shape[0]
            group3_hits = best_hits_df[best_hits_df['prize'] == '组选3'].shape[0]  
            group6_hits = best_hits_df[best_hits_df['prize'] == '组选6'].shape[0]
            logger.info(f"    - 直选(三位全对): {direct_hits} / {num_periods_tested} 期")
            logger.info(f"    - 组选3(三数字对,有重复): {group3_hits} / {num_periods_tested} 期")
            logger.info(f"    - 组选6(三数字对,无重复): {group6_hits} / {num_periods_tested} 期")
    else: logger.warning("回测未产生有效结果，可能是数据量不足。")
    
    # 8. 最终推荐
    logger.info("\n" + "="*25 + f" 第 {last_period + 1} 期 号 码 推 荐 " + "="*25)
    final_recs, final_rec_strings, analysis_summary, _, final_scores = run_analysis_and_recommendation(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules)
    
    logger.info("\n--- 单式推荐 ---")
    for line in final_rec_strings: logger.info(line)
    
    logger.info("\n--- 复式参考 ---")
    if final_scores and final_scores.get('red_1'):
        # 获取各位置的高分号码
        top_pos1 = sorted([n for n, _ in sorted(final_scores['red_1'].items(), key=lambda x: x[1], reverse=True)[:5]])
        top_pos2 = sorted([n for n, _ in sorted(final_scores['red_2'].items(), key=lambda x: x[1], reverse=True)[:5]])
        top_pos3 = sorted([n for n, _ in sorted(final_scores['red_3'].items(), key=lambda x: x[1], reverse=True)[:5]])
        logger.info(f"  百位 (Top 5): {' '.join(str(n) for n in top_pos1)}")
        logger.info(f"  十位 (Top 5): {' '.join(str(n) for n in top_pos2)}")
        logger.info(f"  个位 (Top 5): {' '.join(str(n) for n in top_pos3)}")
    
    # 新增：形态分析信息
    if analysis_summary and 'patterns' in analysis_summary:
        form_patterns = analysis_summary['patterns'].get('form_patterns', {})
        if form_patterns:
            logger.info(f"\n--- 形态分析 ---")
            logger.info(f"  历史组三占比: {form_patterns.get('group3_ratio', 0)*100:.1f}%")
            logger.info(f"  历史组六占比: {form_patterns.get('group6_ratio', 0)*100:.1f}%")
            logger.info(f"  历史豹子占比: {form_patterns.get('triple_ratio', 0)*100:.1f}%")
            
            # 分析当前推荐的形态分布
            if final_recs:
                rec_forms = {'group3': 0, 'group6': 0, 'triple': 0}
                for rec in final_recs:
                    unique_count = len(set(rec['numbers']))
                    if unique_count == 2:
                        rec_forms['group3'] += 1
                    elif unique_count == 3:
                        rec_forms['group6'] += 1
                    else:
                        rec_forms['triple'] += 1
                
                total_recs = len(final_recs)
                logger.info(f"  本期推荐形态分布:")
                logger.info(f"    组三: {rec_forms['group3']}注 ({rec_forms['group3']/total_recs*100:.1f}%)")
                logger.info(f"    组六: {rec_forms['group6']}注 ({rec_forms['group6']/total_recs*100:.1f}%)")
                if rec_forms['triple'] > 0:
                    logger.info(f"    豹子: {rec_forms['triple']}注 ({rec_forms['triple']/total_recs*100:.1f}%)")
    
    logger.info("\n" + "="*60 + f"\n--- 报告结束 (详情请查阅: {os.path.basename(log_filename)}) ---\n")
    
    # 9. 微信推送
    try:
        from pls_wxpusher import send_analysis_report, send_wxpusher_message_fallback
        logger.info("正在发送微信推送...")
        
        # 准备复式参考数据
        duplex_reference = None
        if final_scores and final_scores.get('red_1'):
            top_pos1 = sorted([n for n, _ in sorted(final_scores['red_1'].items(), key=lambda x: x[1], reverse=True)[:5]])
            top_pos2 = sorted([n for n, _ in sorted(final_scores['red_2'].items(), key=lambda x: x[1], reverse=True)[:5]])
            top_pos3 = sorted([n for n, _ in sorted(final_scores['red_3'].items(), key=lambda x: x[1], reverse=True)[:5]])
            duplex_reference = {
                'pos1': top_pos1,
                'pos2': top_pos2,
                'pos3': top_pos3
            }
        
        # 发送分析报告推送
        push_result = send_analysis_report(
            report_content="",  # 可以传入完整报告内容
            period=last_period + 1,
            recommendations=final_rec_strings,
            optuna_summary=optuna_summary,
            backtest_stats=backtest_stats,
            duplex_reference=duplex_reference
        )
        
        if push_result.get('success'):
            logger.info("微信推送发送成功")
        else:
            logger.warning(f"微信推送发送失败: {push_result.get('error', '未知错误')}")
            
            # 如果主要方法失败，尝试备用方法
            logger.info("尝试使用备用推送方法...")
            try:
                # 构建简化的推送内容
                simple_content = f"🎯 排列三第{last_period + 1}期预测报告\n\n📋 推荐号码:\n"
                for i, rec in enumerate(final_rec_strings[:5]):  # 只发送前5注
                    simple_content += f"第{i+1}注: {rec}\n"
                simple_content += "\n⚠️ 详细分析请查看系统日志"
                
                fallback_result = send_wxpusher_message_fallback(
                    content=simple_content,
                    title=f"🎯 排列三第{last_period + 1}期预测 (备用推送)"
                )
                
                if fallback_result.get('success'):
                    logger.info("备用微信推送发送成功")
                else:
                    logger.warning(f"备用微信推送也失败: {fallback_result.get('error', '未知错误')}")
                    
            except Exception as fallback_e:
                logger.error(f"备用微信推送异常: {fallback_e}")
            
    except ImportError:
        logger.warning("微信推送模块未找到，跳过推送功能")
    except Exception as e:
        logger.error(f"微信推送发送异常: {e}")
    
    # 10. 更新latest_pls_analysis.txt
    try:
        latest_file_path = os.path.join(SCRIPT_DIR, 'latest_pls_analysis.txt')
        
        # 检查日志文件是否存在且有内容
        if not os.path.exists(log_filename):
            logger.error(f"日志文件不存在: {log_filename}")
            # 创建一个基本的报告文件
            with open(latest_file_path, 'w', encoding='utf-8') as f:
                f.write(f"排列三分析报告\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"预测期号: 第{last_period + 1}期\n")
                f.write(f"状态: 日志文件生成失败\n")
        else:
            log_file_size = os.path.getsize(log_filename)
            logger.info(f"日志文件大小: {log_file_size} 字节")
            
            if log_file_size == 0:
                logger.warning("日志文件为空，创建基本报告")
                with open(latest_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"排列三分析报告\n")
                    f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"预测期号: 第{last_period + 1}期\n")
                    f.write(f"状态: 分析完成但日志为空\n")
                    if final_rec_strings:
                        f.write(f"\n推荐号码:\n")
                        for i, rec in enumerate(final_rec_strings):
                            f.write(f"注 {i+1}: {rec}\n")
            else:
                # 复制完整的日志文件内容
                with open(log_filename, 'r', encoding='utf-8') as log_f:
                    content = log_f.read()
                    
                with open(latest_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        # 验证生成的文件
        if os.path.exists(latest_file_path):
            final_file_size = os.path.getsize(latest_file_path)
            logger.info(f"已更新 latest_pls_analysis.txt (大小: {final_file_size} 字节)")
            
            # 在CI环境中输出文件内容的前几行用于调试
            if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
                logger.info("=== CI环境调试信息 ===")
                try:
                    with open(latest_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:10]  # 读取前10行
                        logger.info(f"latest_pls_analysis.txt 前10行内容:")
                        for i, line in enumerate(lines, 1):
                            logger.info(f"第{i}行: {line.strip()}")
                except Exception as debug_e:
                    logger.error(f"读取文件用于调试失败: {debug_e}")
        else:
            logger.error("latest_pls_analysis.txt 文件创建失败")
            
    except Exception as e:
        logger.error(f"更新 latest_pls_analysis.txt 失败: {e}")
        # 尝试创建一个最小的报告文件
        try:
            emergency_path = os.path.join(SCRIPT_DIR, 'latest_pls_analysis.txt')
            with open(emergency_path, 'w', encoding='utf-8') as f:
                f.write(f"排列三分析报告 (紧急模式)\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"预测期号: 第{last_period + 1}期\n")
                f.write(f"错误信息: {str(e)}\n")
            logger.info("已创建紧急模式报告文件")
        except Exception as emergency_e:
            logger.critical(f"紧急模式报告文件创建也失败: {emergency_e}")


def save_analysis_report(df: pd.DataFrame, freq_data: Dict, pattern_data: Dict, 
                        recommendations: List[Dict], details: List[str], period: int):
    """保存分析报告到文件"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(SCRIPT_DIR, f"pls_analysis_output_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("排列三数据分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"预测期号: 第{period}期\n")
            f.write(f"分析基于数据: 截至 {int(df['Seq'].iloc[-1])} 期\n")
            f.write(f"历史数据量: {len(df)} 期\n\n")
            
            # 推荐组合
            f.write("推荐组合:\n")
            f.write("-" * 30 + "\n")
            for i, combo in enumerate(recommendations):
                f.write(f"注 {i+1}: [{combo['numbers'][0]}, {combo['numbers'][1]}, {combo['numbers'][2]}]\n")
            
            f.write(f"\n详细评分:\n")
            f.write("-" * 30 + "\n")
            for detail in details:
                f.write(f"{detail}\n")
            
            # 模式分析摘要
            f.write(f"\n模式分析摘要:\n")
            f.write("-" * 30 + "\n")
            f.write(f"常见奇数个数: {pattern_data.get('odd_patterns', {}).get('most_common', '未知')}\n")
            f.write(f"常见大数个数: {pattern_data.get('big_patterns', {}).get('most_common', '未知')}\n")
            f.write(f"平均和值: {pattern_data.get('sum_patterns', {}).get('avg_sum', 0):.1f}\n")
            f.write(f"平均跨度: {pattern_data.get('span_patterns', {}).get('avg_span', 0):.1f}\n")
        
        logger.info(f"分析报告已保存: {report_file}")
        
    except Exception as e:
        logger.error(f"保存分析报告失败: {e}")


# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    # 设置控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 运行主程序
    main()
