import json
import os
import sys
import traceback
from typing import List, Dict, Any

# 假设 QwenAgent_vllm 和 Constant 位于您的 Python 路径中
# 并且 QwenVllmAgent 已经设置好
try:
    from QwenAgent_vllm import QwenVllmAgent
    from Constant import QWEN_2_5_AGENT_7B_NAME
except ImportError:
    print("错误: 无法导入 'QwenAgent_vllm' 或 'Constant'。")
    print("请确保这些模块位于您的 PYTHONPATH 中。")
    # --- 为本地测试提供 Mock 对象 ---
    print("... 使用 Mock 对象继续执行以进行演示 ...")
    class MockQwenVllmAgent:
        def __init__(self, model_name, system_prompt, is_api, max_new_tokens):
            print(f"[Mock] QwenVllmAgent 初始化 (model={model_name})")
            print(f"[Mock] System Prompt: {system_prompt[:100]}...")
            self.system_prompt = system_prompt
            self.max_tokens = max_new_tokens

        def query(self, prompts: List[str]) -> List[str]:
            print(f"[Mock] 收到 {len(prompts)} 个提示词的批量查询")
            responses = []
            for p in prompts:
                # 确保 p 是字符串 (模拟修复)
                p = str(p) 
                if "2018-10-31" in p:
                    responses.append("2018-10-31")
                elif "2023-11-16" in p:
                    responses.append("2023-11-16")
                elif "2021-03-15" in p:
                     responses.append("2021-03-15")
                elif "April 27, 2021" in p:
                     responses.append("2021-04-27")
                elif "August 2" in p or "Nov. 12" in p:
                    responses.append("undefined") # 模拟无法解析
                else:
                    responses.append("undefined")
            print(f"[Mock] 返回 {len(responses)} 个响应")
            return responses
    
    QwenVllmAgent = MockQwenVllmAgent
    QWEN_2_5_AGENT_7B_NAME = "mock-qwen-7b"
    # --- 结束 Mock 对象 ---

def run_timestamp_unification_pipeline():
    """
    运行VLLM管道，以统一化JSON文件中的新闻时间戳。
    它会加载现有数据，批量处理所有缺少统一时间戳的条目，
    并保存到一个新文件中。
    """
    
    # 1. 定义新的系统提示 (专门用于日期统一化)
    system_prompt_date = """
You are an expert data cleaning and timestamp normalization assistant.
Your task is to analyze the provided 'original_timestamp' string and convert it into a single, unified date in `YYYY-MM-DD` format.

Follow these rules STRICTLY:
1.  If the string provides a full, unambiguous date (e.g., '2023-10-15T12:00:00Z', 'Nov. 16, 2018', 'First Published: Oct 31 2018 | 8:57 AM IST'), extract and format it as `YYYY-MM-DD`.
2.  If the string is ambiguous and **lacks a specific year** (e.g., 'August 2', 'Friday, August 31', 'Nov. 12', 'Wednesday, October 3'), you MUST respond with the exact string `undefined`.
3.  If the string is completely unusable or vague (e.g., 'Thursday at the Yahoo Finance All Markets Summit', 'Saturday evening', 'Thursday (date not specified)'), you MUST respond with the exact string `undefined`.

Respond ONLY with the resulting string (e.g., `2018-10-31` or `undefined`). Do not add any other text, explanations, or JSON formatting.
"""

    # 2. 初始化 QwenVllmAgent
    # (max_new_tokens 可以很低, 因为我们只需要一个日期或 'undefined')
    agent = QwenVllmAgent(
        model_name=QWEN_2_5_AGENT_7B_NAME,
        system_prompt=system_prompt_date,
        is_api=False,
        max_new_tokens=20 # 日期格式 YYYY-MM-DD (10) 或 undefined (9)
    )

    # 3. 指定输入和输出文件路径
    input_file_path = "/scratch/u5ci/liuhongyuan.u5ci/timellm/time_agent/bitcoin_news_analysis_results.json"
    output_file_path = "/scratch/u5ci/liuhongyuan.u5ci/timellm/time_agent/bitcoin_news_analysis_results_unified.json"
    
    # 4. 加载源数据 (必须存在)
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            source_articles = json.load(f)
        print(f"成功加载 {len(source_articles)} 条源数据 (来自 {input_file_path})")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载源文件: {e}")
        print("管道已停止。")
        return

    # 5. [Checkpointing] 加载已有的统一化结果 (如果存在)
    # 我们使用 'article_link' 作为唯一ID来跟踪已处理的条目
    processed_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    existing_results = json.loads(content)
                    for item in existing_results:
                        link = item.get('article_link')
                        # 我们只关心那些 *已经* 具有统一字段的条目
                        if link and 'news_timestamp_unified' in item:
                            processed_map[link] = item
            print(f"成功加载 {len(processed_map)} 条已处理的条目 (来自 {output_file_path})")
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法读取现有的统一化文件 '{output_file_path}'. 将重新处理所有内容. 错误: {e}")
            processed_map = {}

    # 6. 收集所有需要处理的条目
    articles_to_process: List[Dict[str, Any]] = []
    prompts_to_process: List[str] = []
    final_results_list: List[Dict[str, Any]] = [] # 这将保存所有条目

    print(f"\n正在过滤 {len(source_articles)} 条源数据以查找未处理的时间戳...")
    
    for article in source_articles:
        article_link = article.get('article_link')
        
        # 检查是否已在之前的运行中处理过
        if article_link and article_link in processed_map:
            # 已处理, 直接使用旧结果
            final_results_list.append(processed_map[article_link])
        else:
            # 未处理, 将其添加到批量处理中
            original_timestamp = article.get('news_timestamp', '')
            if not original_timestamp:
                # 如果没有时间戳, 标记为 undefined 并添加到最终列表, 而不查询 LLM
                article['news_timestamp_unified'] = 'undefined'
                final_results_list.append(article)
                continue
            
            # *** 错误修复 ***
            # 确保时间戳是一个字符串，以防止 'int' object is not subscriptable 错误
            if not isinstance(original_timestamp, str):
                original_timestamp = str(original_timestamp)
            # *** 修复结束 ***
                
            articles_to_process.append(article)
            # *** 更改：添加明确的提示词 ***
            # 仅仅发送原始日期字符串可能会让模型困惑。
            # 我们添加一个包装器来激活系统提示。
            user_prompt = f"Here is the original_timestamp string to analyze: {original_timestamp}"
            prompts_to_process.append(user_prompt)
            # *** 更改结束 ***

    # 7. 批量处理所有新条目
    if not prompts_to_process:
        print("没有找到新的时间戳需要统一化。")
    else:
        print(f"\n=============== 正在批量处理 {len(prompts_to_process)} 个新时间戳 ===============")
        
        try:
            # 使用 VLLM 批量查询
            batch_responses = agent.query(prompts_to_process)
            
            newly_processed_count = 0
            # 8. 处理响应并将其合并
            for article, unified_date_response in zip(articles_to_process, batch_responses):
                if not unified_date_response or not isinstance(unified_date_response, str):
                    print(f"警告: 收到无效响应 (非字符串) (原文: {article.get('news_timestamp')}). 标记为 undefined.")
                    unified_date = 'undefined'
                else:
                    unified_date = unified_date_response.strip()

                # (可选) 对 LLM 的输出进行简单验证
                if not (len(unified_date) == 10 and unified_date[4] == '-' and unified_date[7] == '-') and unified_date != 'undefined':
                    print(f"警告: LLM 返回了非标准格式: '{unified_date}' (原文: {article.get('news_timestamp')}). 强制设为 'undefined'.")
                    unified_date = 'undefined'
                
                # 添加新字段
                article['news_timestamp_unified'] = unified_date
                # 添加到我们的最终列表
                final_results_list.append(article)
                newly_processed_count += 1
            
            print(f"--- 成功统一化 {newly_processed_count}/{len(prompts_to_process)} 个时间戳 ---")

        except Exception as e:
            print(f"\n!!! VLLM 批量查询期间发生严重错误: {e} !!!")
            print(traceback.format_exc())
            print("--- 正在跳过此批次 ---")

    # 9. [保存结果] 将包含所有条目(旧的和新的)的完整列表写入新文件
    try:
        # 按原始时间戳排序 (可选, 但有助于保持一致性)
        # 修复: 确保排序键是字符串
        final_results_list.sort(key=lambda x: str(x.get('news_timestamp', '')))
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_list, f, indent=4, ensure_ascii=False)
        print(f"\n--- 所有 {len(final_results_list)} 条结果已保存到 '{output_file_path}' ---")
    except IOError as e:
        print(f"!!! 严重错误: 无法将最终结果保存到 '{output_file_path}'. 错误: {e} !!!")
    except TypeError as e:
        # 捕获排序时可能发生的类型错误 (例如比较 int 和 str)
        print(f"!!! 严重错误: 保存前排序失败: {e} !!!")
        print("... 正在尝试不排序直接保存 ...")
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=4, ensure_ascii=False)
            print(f"\n--- 所有 {len(final_results_list)} 条结果已(未排序)保存到 '{output_file_path}' ---")
        except IOError as e_save:
            print(f"!!! 严重错误: 尝试不排序保存也失败了: {e_save} !!!")


    print("\n\n======== 时间戳统一化任务完成 ========")


if __name__ == "__main__":
    run_timestamp_unification_pipeline()

