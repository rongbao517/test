# 模块说明：该脚本用于批量分析新闻对比特币价格的潜在影响，使用 QwenVllmAgent（本地 vLLM 版本）并支持断点续跑。
# 依赖：
#   - QwenAgent.py / QwenAgent_vllm.py：自定义的代理类，用于与 Qwen 模型交互
#   - Constant.py：常量定义，包括模型名称等
#   - torch, numpy, json：标准库和第三方库，用于数据处理和模型推理

import json
import traceback
import random
import os
from Constant import *
import numpy as np
import torch
from QwenAgent import QwenAgent
from QwenAgent_vllm import QwenVllmAgent
# # --- Mock Dependencies and Placeholders ---
# # NOTE: These are mock objects to make the script runnable without the actual
# # proprietary libraries or API keys. Replace these with your actual implementations.


# # Mocking the OpenAI API client and its response structure
# class MockChoice:
#     def __init__(self, content):
#         self.message = type('obj', (object,), {'content': content})

# class MockCompletion:
#     def __init__(self, content):
#         self.choices = [MockChoice(content)]

# class MockChat:
#     def completions_create(self, model, messages, temperature, seed, top_p, frequency_penalty, presence_penalty):
#         # Simulate the model's response based on the prompt.
#         # A real model would generate this JSON content.
#         mock_response_content = {
#             "relevance_to_bitcoin_price": round(random.uniform(0.1, 0.9), 2),
#             "impact_analysis": {
#                 "short_term_impact_5_days": {
#                     "effect": "rise" if random.random() > 0.5 else "fall",
#                     "percentage_change": f"{'+' if random.random() > 0.5 else '-'}{random.randint(1, 15)}%"
#                 },
#                 "medium_term_impact_15_days": {
#                     "effect": "rise" if random.random() > 0.5 else "fall",
#                     "percentage_change": f"{'+' if random.random() > 0.5 else '-'}{random.randint(5, 25)}%"
#                 },
#                 "long_term_impact_after_15_days": {
#                     "effect": "rise" if random.random() > 0.5 else "fall",
#                     "percentage_change": f"{'+' if random.random() > 0.5 else '-'}{random.randint(10, 50)}%"
#                 }
#             },
#             "news_timestamp": "2018-01-01 00:00:00" # This would be extracted from the article
#         }
#         return MockCompletion(json.dumps(mock_response_content, indent=2))

# class MockOpenAI:
#     def __init__(self, api_key, base_url, timeout):
#         self.chat = MockChat()
#         print("MockOpenAI client initialized.")

# # Mocking json_repair library
# class MockJsonRepair:
#     def loads(self, text):
#         return json.loads(text)

# json_repair = MockJsonRepair()

# # Mocking torch and transformers for the non-API path
# class MockTokenizer:
#     def apply_chat_template(self, messages, tokenize, add_generation_prompt):
#         return f"Simulated template for: {messages[-1]['content'][:100]}..."
#     def __call__(self, text, return_tensors):
#         return type('obj', (object,), {'input_ids': [1,2,3], 'to': lambda x: self})
#     def batch_decode(self, ids, skip_special_tokens):
#         return ["mock response from local model"]

# class MockModel:
#     def generate(self, **kwargs):
#         return [[1,2,3,4,5,6]]
#     def to(self, device):
#         return self
#     def set_attn_implementation(self, implementation):
#         pass

# # Since the user's code uses these, we define them.
# AutoModelForCausalLM = type('obj', (object,), {'from_pretrained': lambda *args, **kwargs: MockModel()})
# AutoTokenizer = type('obj', (object,), {'from_pretrained': lambda *args, **kwargs: MockTokenizer()})
# torch = type('obj', (object,), {'no_grad': __import__('contextlib').contextmanager(lambda: (yield)), 'torch_dtype': str})







# --- Main Pipeline Execution ---

def run_parallel_analysis_pipeline():
    """
    运行并行新闻分析流水线（中文说明）。

    功能概述：
      - 读取原始新闻文件（json 列表），挑选尚未分析的文章。
      - 为每篇待处理文章构造分析 prompt，并使用 QwenVllmAgent 批量查询（并行/一次性）。
      - 将模型返回的结构化结果与文章元信息合并，支持断点续跑（已存在的结果会被跳过）。
      - 将新增结果追加并保存到输出文件。

    注意：
      - 本函数只负责 orchestration（协调流程），实际的语言模型调用由 QwenVllmAgent.query 实现。
      - 输出文件保存为 JSON 列表，单条记录包含至少 article_link 与结构化分析字段，便于后续审阅与复现。
    """
    # 1. 定义系统提示（system prompt），用于引导模型输出固定结构的 JSON 结果
    system_prompt_en = """
    You are an expert financial analyst specializing in the cryptocurrency market.
    please analyze the impact of news articles on Bitcoin's price movements and why, then provide a structured JSON response:
    - "relevance_to_bitcoin_price": A float between 0.0 and 1.0 indicating how relevant the news is to Bitcoin's price. (e.g., a domestic Indian news might be 0.1, a US Fed decision might be 0.9).
    - "impact_analysis": An object containing predictions for Bitcoin's price.
        - "short_term_impact_5_days": An object with "effect" ('rise' or 'fall') and "percentage_change" (e.g., "+0.15").
        - "medium_term_impact_15_days": An object with "effect" and "percentage_change".
        - "long_term_impact_after_15_days": An object with "effect" and "percentage_change".
    - "news_timestamp": The publication time from the news data.
    Analyze carefully and provide a reasoned forecast.
    """

    # 2. 初始化 agent（本地 vLLM 或 API），is_api=False 表示走本地推理路径
    agent = QwenVllmAgent(
        model_name=QWEN_2_5_AGENT_7B_NAME,
        system_prompt=system_prompt_en,
        is_api=False,
        max_new_tokens=512
    )

    # 3. 指定输入/输出路径（可按需修改）
    input_file_path = "/scratch/u5ci/liuhongyuan.u5ci/timellm/From_News_to_Forecast/data/raw_news_data/bitcoin_news.json"
    output_file_path = "./bitcoin_news_analysis_results.json"
    
    # 如果输入文件不存在，生成一个示例文件以便演示（实际使用时应确保提供真实数据）
    if not os.path.exists(input_file_path):
        print(f"Warning: File not found at '{input_file_path}'. Creating a dummy file for demonstration.")
        dummy_data = [
            {'title': 'Bitcoin vs gold: Safe haven battle', 'category': '', 'summary': '', 'link': 'https://www.fxstreet.com/cryptocurrencies/news/bitcoin-vs-gold-safe-haven-battle-202010261034', 'publication_time': '2018-01-01 00:00:00', 'full_article': 'A new idea has been floating around in Safe Haven Trading...'},
            {'title': 'Intel Leaks Details On New CPU', 'category': '', 'summary': '', 'link': 'https://www.extremetech.com/gaming/261297-intel-leaks-details-new-desktop-core-i7-8809g-radeon-graphics', 'publication_time': '2018-01-02 08:15:16', 'full_article': 'Ever since Intel and AMD confirmed...'},
            {'title': 'US Fed Considers Rate Hike', 'category': 'Finance', 'summary': '', 'link': 'https://example.com/fed-rate-hike', 'publication_time': '2018-01-03 14:00:00', 'full_article': 'The US Federal Reserve is considering a new rate hike to combat inflation...'}
        ]
        os.makedirs(os.path.dirname(input_file_path) or '.', exist_ok=True)
        with open(input_file_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=4)
            
    # 4. 读取新闻数据（JSON list），并进行简单的错误处理
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Failed to load news data: {e}")
        return

    # 5. 断点续跑：加载已有结果并记录已处理的文章链接集合（processed_links）
    processed_links = set()
    all_results = []
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    all_results = json.loads(content)
                    for result in all_results:
                        if 'article_link' in result:
                            processed_links.add(result['article_link'])
            print(f"Successfully loaded {len(processed_links)} existing analysis results.")
        except (json.JSONDecodeError, IOError) as e:
            # 如果旧文件损坏或无法读取，则警告并从头开始
            print(f"Warning: Could not read existing results file '{output_file_path}'. Starting fresh. Error: {e}")
            all_results = []

    # 6. 筛选出需要处理的文章（跳过已处理或缺少正文的条目），同时构造用于模型的 prompt 列表
    articles_to_process = []
    prompts_to_process = []
    print(f"\nFound {len(articles)} total articles. Filtering for new articles to process...")
    for article in articles:
        article_link = article.get('link', '')
        # 跳过无效条目或已处理文章（checkpoint）
        if not article_link or not article.get('full_article', ''):
            continue
        if article_link in processed_links:
            continue
        
        # 为该文章生成分析请求的用户 prompt（将 title + full_article 作为内容）
        user_prompt = f"Please analyze the following news article:\n\nTitle: {article.get('title', 'No Title')}\n\nContent: {article.get('full_article', '')}"
        articles_to_process.append(article)
        prompts_to_process.append(user_prompt)

    # 7. 如果有待处理的 prompt，则使用 agent 批量查询（一次性传入所有 prompt）
    if not prompts_to_process:
        print("No new articles to analyze. All tasks are complete.")
    else:
        print(f"\n=============== Processing a batch of {len(prompts_to_process)} new articles ===============")
        
        # 将整个 prompt 列表传给 agent.query，agent 负责并行/批量化的实现细节
        batch_results = agent.query(prompts_to_process)

        # 8. 将模型返回的结构化分析结果与原始文章属性合并，便于保存和后续审阅
        newly_processed_count = 0
        for article, analysis_result in zip(articles_to_process, batch_results):
            if analysis_result and isinstance(analysis_result, dict):
                # 为可追溯性添加文章元数据（标题与链接）
                analysis_result['news_title'] = article.get('title', 'No Title')
                analysis_result['article_link'] = article.get('link')
                all_results.append(analysis_result)
                newly_processed_count += 1
            else:
                # 若某条分析失败，打印警告（但继续处理其他条目）
                print(f"\n--- Analysis Failed ---")
                print(f"Could not get a valid structured response for article: '{article.get('title', 'No Title')}'")
                print("-----------------------\n")
        
        print(f"--- Successfully analyzed {newly_processed_count}/{len(prompts_to_process)} articles in the batch ---")

        # 9. 如果有新增分析结果则保存回 JSON 文件（覆写旧文件）
        if newly_processed_count > 0:
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)
                print(f"--- Results have been saved to '{output_file_path}'")
            except IOError as e:
                # 保存失败属于关键错误，需要关注磁盘权限/空间等问题
                print(f"!!! CRITICAL ERROR: Could not save results to '{output_file_path}'. Error: {e} !!!")

    print("\n\n======== All analysis tasks are complete ========")


if __name__ == "__main__":
    run_parallel_analysis_pipeline()
