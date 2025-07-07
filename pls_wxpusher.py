# -*- coding: utf-8 -*-
"""
排列三微信推送模块
================

提供微信推送功能，用于推送排列三分析报告和验证报告
"""

import requests
import logging
import json
import os
from datetime import datetime
from typing import Optional, List, Dict

# 微信推送配置
# 支持从环境变量读取配置（用于GitHub Actions等CI环境）
APP_TOKEN = os.getenv("WXPUSHER_APP_TOKEN", "AT_FInZJJ0mUU8xvQjKRP7v6omvuHN3Fdqw")
USER_UIDS = os.getenv("WXPUSHER_USER_UIDS", "UID_yYObqdMVScIa66DGR2n2PCRFL10w").split(",")
TOPIC_IDS = [int(x) for x in os.getenv("WXPUSHER_TOPIC_IDS", "39909").split(",") if x.strip()]

def get_latest_verification_result() -> Optional[Dict]:
    """获取最新的验证结果
    
    Returns:
        最新验证结果字典，如果没有则返回None
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calc_file = os.path.join(script_dir, 'latest_pls_calculation.txt')
        
        if not os.path.exists(calc_file):
            return None
            
        with open(calc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析最新的验证记录
        lines = content.split('\n')
        
        # 查找最新的评估记录
        for i, line in enumerate(lines):
            if line.startswith('评估时间:'):
                # 解析评估信息
                result = {}
                
                # 解析期号
                for j in range(i, min(i+20, len(lines))):
                    if lines[j].startswith('评估期号'):
                        result['eval_period'] = lines[j].split(':')[1].strip().split()[0]
                    elif lines[j].startswith('开奖号码:'):
                        # 解析开奖号码
                        draw_line = lines[j]
                        try:
                            number_str = lines[j].split(':')[1].strip()
                            if len(number_str) == 3:
                                result['prize_numbers'] = [int(number_str[0]), int(number_str[1]), int(number_str[2])]
                        except:
                            pass
                    elif lines[j].startswith('总奖金:'):
                        try:
                            amount_str = lines[j].split(':')[1].strip().replace('元', '').replace(',', '')
                            result['total_prize'] = int(amount_str) if amount_str.isdigit() else 0
                        except:
                            result['total_prize'] = 0
                
                return result if result else None
                
        return None
        
    except Exception as e:
        logging.error(f"获取最新验证结果失败: {e}")
        return None

def send_wxpusher_message(content: str, title: str = None, topicIds: List[int] = None, uids: List[str] = None) -> Dict:
    """
    发送微信推送消息
    
    Args:
        content: 消息内容
        title: 消息标题
        topicIds: 主题ID列表，默认使用全局配置
        uids: 用户ID列表，默认使用全局配置
    
    Returns:
        API响应结果字典
    """
    import urllib3
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    # 禁用SSL警告
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    url = "https://wxpusher.zjiecode.com/api/send/message"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    data = {
        "appToken": APP_TOKEN,
        "content": content,
        "uids": uids or USER_UIDS,
        "topicIds": topicIds or TOPIC_IDS,
        "summary": title or "排列三推荐更新",
        "contentType": 1,  # 1=文本，2=HTML
    }
    
    # 配置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # 创建会话并配置适配器
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        # 尝试多种方式发送请求
        for attempt in range(3):
            try:
                if attempt == 0:
                    # 第一次尝试：正常HTTPS请求
                    response = session.post(url, json=data, headers=headers, timeout=30, verify=True)
                elif attempt == 1:
                    # 第二次尝试：禁用SSL验证
                    response = session.post(url, json=data, headers=headers, timeout=30, verify=False)
                else:
                    # 第三次尝试：使用HTTP（如果服务支持）
                    http_url = url.replace('https://', 'http://')
                    response = session.post(http_url, json=data, headers=headers, timeout=30)
                
                response.raise_for_status()
                result = response.json()
                
                if result.get("success", False):
                    logging.info(f"微信推送成功: {title} (尝试次数: {attempt + 1})")
                    return {"success": True, "data": result}
                else:
                    logging.error(f"微信推送失败: {result.get('msg', '未知错误')}")
                    return {"success": False, "error": result.get('msg', '推送失败')}
                    
            except requests.exceptions.SSLError as ssl_e:
                logging.warning(f"SSL错误 (尝试 {attempt + 1}/3): {ssl_e}")
                if attempt == 2:  # 最后一次尝试也失败
                    raise ssl_e
                continue
            except requests.exceptions.RequestException as req_e:
                logging.warning(f"网络请求错误 (尝试 {attempt + 1}/3): {req_e}")
                if attempt == 2:  # 最后一次尝试也失败
                    raise req_e
                continue
                
    except requests.exceptions.SSLError as e:
        error_msg = f"SSL连接错误: {str(e)}。建议检查网络环境或联系管理员。"
        logging.error(f"微信推送SSL错误: {error_msg}")
        return {"success": False, "error": error_msg}
    except requests.exceptions.RequestException as e:
        error_msg = f"网络连接错误: {str(e)}"
        logging.error(f"微信推送网络错误: {error_msg}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"未知异常: {str(e)}"
        logging.error(f"微信推送异常: {error_msg}")
        return {"success": False, "error": error_msg}
    finally:
        session.close()

def send_analysis_report(report_content: str, period: int, recommendations: List[str], 
                         optuna_summary: Dict = None, backtest_stats: Dict = None, 
                         duplex_reference: Dict = None) -> Dict:
    """发送排列三分析报告
    
    Args:
        report_content: 完整的分析报告内容
        period: 预测期号
        recommendations: 推荐号码列表
        optuna_summary: Optuna优化摘要
        backtest_stats: 回测统计数据
        duplex_reference: 复式参考数据 {'pos1': [1,2,3,4,5], 'pos2': [0,1,4,6,9], 'pos3': [1,4,6,7,9]}
    
    Returns:
        推送结果字典
    """
    title = f"🎯 排列三第{period}期预测报告"
    
    # 提取关键信息制作详细版推送
    try:
        # 获取最新验证结果
        latest_verification = get_latest_verification_result()
        
        # 构建推荐内容 - 显示所有推荐号码，采用紧凑格式
        rec_summary = ""
        if recommendations:
            for i, rec in enumerate(recommendations):
                # 提取号码部分，简化格式
                import re
                number_match = re.search(r'\[([0-9\s,]+)\]', rec)
                
                if number_match:
                    # 格式化为三位数字
                    numbers_str = number_match.group(1)
                    numbers = [x.strip() for x in re.findall(r'\d', numbers_str)]
                    if len(numbers) >= 3:
                        formatted_number = f"{numbers[0]}{numbers[1]}{numbers[2]}"
                        rec_summary += f"第{i+1:2d}注: {formatted_number}\n"
                else:
                    # 如果解析失败，使用原始格式但简化
                    rec_summary += f"第{i+1:2d}注: {rec}\n"
        
        # 构建优化信息
        optuna_info = ""
        if optuna_summary and optuna_summary.get('status') == '完成':
            optuna_info = f"🔬 Optuna优化得分: {optuna_summary.get('best_value', 0):.2f}\n"
        
        # 构建回测信息
        backtest_info = ""
        if backtest_stats:
            prize_counts = backtest_stats.get('prize_counts', {})
            if prize_counts:
                prize_info = []
                for prize, count in prize_counts.items():
                    if count > 0:
                        prize_info.append(f"{prize}x{count}")
                if prize_info:
                    backtest_info = f"📊 最近回测: {', '.join(prize_info)}\n"
        
        # 构建上期验证信息
        verification_info = ""
        if latest_verification:
            eval_period = latest_verification.get('eval_period', '未知')
            prize_nums = latest_verification.get('prize_numbers', [])
            total_prize = latest_verification.get('total_prize', 0)
            
            if prize_nums:
                prize_str = f"{prize_nums[0]}{prize_nums[1]}{prize_nums[2]}"
                if total_prize > 0:
                    verification_info = f"💰 第{eval_period}期验证: 开奖{prize_str}，中奖{total_prize}元\n"
                else:
                    verification_info = f"📈 第{eval_period}期验证: 开奖{prize_str}，未中奖\n"
        
        # 构建复式参考信息
        duplex_info = ""
        if duplex_reference:
            pos1_nums = ' '.join(str(n) for n in duplex_reference.get('pos1', []))
            pos2_nums = ' '.join(str(n) for n in duplex_reference.get('pos2', []))
            pos3_nums = ' '.join(str(n) for n in duplex_reference.get('pos3', []))
            duplex_info = f"""🎲 复式参考:
• 百位: {pos1_nums}
• 十位: {pos2_nums}
• 个位: {pos3_nums}

"""
        
        # 组合最终推送内容
        content = f"""🎯 排列三第{period}期预测报告

{verification_info}{optuna_info}{backtest_info}
📋 本期推荐 ({len(recommendations)}注):
{rec_summary}
{duplex_info}💡 投注建议：
• 直选投注：按推荐顺序投注
• 组选投注：可组合投注降低风险
• 复式投注：可根据复式参考组合投注
• 建议控制投注金额，理性投注

📊 分析说明：
基于历史数据统计分析、机器学习预测和关联规则挖掘，综合生成推荐号码。

⚠️ 风险提示：
彩票具有随机性，请理性投注，量力而行。"""

        return send_wxpusher_message(content, title)
        
    except Exception as e:
        logging.error(f"构建分析报告推送内容时出错: {e}")
        return {"success": False, "error": f"构建推送内容失败: {str(e)}"}

def send_verification_report(verification_data: Dict) -> Dict:
    """发送验证报告
    
    Args:
        verification_data: 验证数据字典
    
    Returns:
        推送结果字典
    """
    try:
        period = verification_data.get('eval_period', '未知')
        prize_numbers = verification_data.get('prize_numbers', [])
        total_prize = verification_data.get('total_prize', 0)
        winning_count = verification_data.get('winning_count', 0)
        total_recommendations = verification_data.get('total_recommendations', 0)
        
        title = f"📊 排列三第{period}期验证报告"
        
        if prize_numbers and len(prize_numbers) >= 3:
            prize_str = f"{prize_numbers[0]}{prize_numbers[1]}{prize_numbers[2]}"
        else:
            prize_str = "未知"
        
        if total_prize > 0:
            content = f"""🎉 排列三第{period}期验证报告

📈 开奖结果: {prize_str}
🎯 推荐数量: {total_recommendations}注
💰 中奖情况: {winning_count}注中奖
💵 总奖金: {total_prize}元

🎊 恭喜中奖！预测准确度较高。"""
        else:
            content = f"""📊 排列三第{period}期验证报告

📈 开奖结果: {prize_str}
🎯 推荐数量: {total_recommendations}注
💰 中奖情况: 未中奖
💵 总奖金: 0元

📝 本期未中奖，将继续优化预测算法。"""
        
        return send_wxpusher_message(content, title)
        
    except Exception as e:
        logging.error(f"发送验证报告时出错: {e}")
        return {"success": False, "error": f"发送验证报告失败: {str(e)}"}

def send_error_notification(error_msg: str, script_name: str = "排列三系统") -> Dict:
    """发送错误通知
    
    Args:
        error_msg: 错误信息
        script_name: 脚本名称
    
    Returns:
        推送结果字典
    """
    title = f"⚠️ {script_name}运行异常"
    content = f"""⚠️ {script_name}运行异常

⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
❌ 错误: {error_msg}

请检查系统状态并及时处理。"""
    
    return send_wxpusher_message(content, title)

def send_daily_summary(analysis_success: bool, verification_success: bool, 
                      analysis_file: str = None, error_msg: str = None) -> Dict:
    """发送每日总结
    
    Args:
        analysis_success: 分析是否成功
        verification_success: 验证是否成功
        analysis_file: 分析文件名
        error_msg: 错误信息
    
    Returns:
        推送结果字典
    """
    title = "📈 排列三系统日报"
    
    status_analysis = "✅ 成功" if analysis_success else "❌ 失败"
    status_verification = "✅ 成功" if verification_success else "❌ 失败"
    
    content = f"""📈 排列三系统日报

⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 今日运行状态:
• 数据分析: {status_analysis}
• 结果验证: {status_verification}
"""
    
    if analysis_file:
        content += f"• 分析报告: {analysis_file}\n"
    
    if error_msg:
        content += f"\n❌ 错误信息:\n{error_msg}"
    
    if analysis_success and verification_success:
        content += "\n\n🎉 系统运行正常，所有任务执行成功！"
    else:
        content += "\n\n⚠️ 系统存在异常，请及时检查处理。"
    
    return send_wxpusher_message(content, title)

def send_wxpusher_message_fallback(content: str, title: str = None, topicIds: List[int] = None, uids: List[str] = None) -> Dict:
    """
    备用微信推送方法（使用更简单的方式）
    
    Args:
        content: 消息内容
        title: 消息标题
        topicIds: 主题ID列表，默认使用全局配置
        uids: 用户ID列表，默认使用全局配置
    
    Returns:
        API响应结果字典
    """
    try:
        import urllib.request
        import urllib.parse
        import ssl
        
        # 创建SSL上下文，忽略证书验证
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        url = "https://wxpusher.zjiecode.com/api/send/message"
        
        data = {
            "appToken": APP_TOKEN,
            "content": content,
            "uids": uids or USER_UIDS,
            "topicIds": topicIds or TOPIC_IDS,
            "summary": title or "排列三推荐更新",
            "contentType": 1,
        }
        
        # 转换为JSON字符串
        json_data = json.dumps(data).encode('utf-8')
        
        # 创建请求
        req = urllib.request.Request(
            url,
            data=json_data,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Python-urllib/3.0'
            }
        )
        
        # 发送请求
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            result_data = response.read().decode('utf-8')
            result = json.loads(result_data)
            
            if result.get("success", False):
                logging.info(f"微信推送成功 (备用方法): {title}")
                return {"success": True, "data": result}
            else:
                logging.error(f"微信推送失败 (备用方法): {result.get('msg', '未知错误')}")
                return {"success": False, "error": result.get('msg', '推送失败')}
                
    except Exception as e:
        error_msg = f"备用推送方法失败: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "error": error_msg}


def test_wxpusher_connection() -> bool:
    """
    测试微信推送连接
    
    Returns:
        连接是否成功
    """
    test_content = f"🔔 排列三系统测试消息\n\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n如果您看到此消息，说明推送功能正常工作。"
    
    # 首先尝试主要方法
    result = send_wxpusher_message(test_content, "🔔 排列三系统测试")
    
    if result.get("success", False):
        return True
    
    # 如果主要方法失败，尝试备用方法
    logging.info("主要推送方法失败，尝试备用方法...")
    fallback_result = send_wxpusher_message_fallback(test_content, "🔔 排列三系统测试 (备用)")
    
    return fallback_result.get("success", False)

if __name__ == "__main__":
    # 测试推送功能
    if test_wxpusher_connection():
        print("✅ 微信推送测试成功")
    else:
        print("❌ 微信推送测试失败")