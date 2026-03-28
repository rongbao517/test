import json
import traceback
import random
import os
from Constant import *
import numpy as np
import torch
from QwenAgent import QwenAgent

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

def run_analysis_pipeline():
    """
    运行新闻分析流程的主函数。
    该函数经过优化，可以边运行边保存结果，并支持断点续运行。
    """
    # 1. 定义模型的系统提示（System Prompt）
    system_prompt_en = """
    You are an expert financial analyst specializing in the cryptocurrency market.
    please analyze the impact of news articles on Bitcoin's price movements and why, then provide a structured JSON response:
    - "relevance_to_bitcoin_price": A float between 0.0 and 1.0 indicating how relevant the news is to Bitcoin's price. (e.g., a domestic Indian news might be 0.1, a US Fed decision might be 0.9).
    - "impact_analysis": An object containing predictions for Bitcoin's price.
        - "short_term_impact_5_days": An object with "effect" ('rise' or 'fall') and "percentage_change" (e.g., "+0.15"). TRY NOT TO GIVE NEUTURAL IF THE RELEVANCE IS HIGH.
        - "medium_term_impact_15_days": An object with "effect" and "percentage_change".
        - "long_term_impact_after_15_days": An object with "effect" and "percentage_change".
    - "news_timestamp": The publication time from the news data.
    Analyze carefully and provide a reasoned forecast.
    """

    # 2. 初始化 QwenAgent
    agent = QwenAgent(
        model_name=QWEN_2_5_AGENT_7B_NAME, # 假设的模型名称
        system_prompt=system_prompt_en,
        is_api=False,max_new_tokens=512
    )

    # 3. 指定输入和输出文件路径
    input_file_path = "/scratch/u5ci/liuhongyuan.u5ci/timellm/From_News_to_Forecast/data/raw_news_data/bitcoin_news.json"
    output_file_path = "./bitcoin_news_analysis_results.json"
    
    # 如果输入文件不存在，创建一个用于演示的虚拟文件
    if not os.path.exists(input_file_path):
        print(f"警告: 在 '{input_file_path}' 未找到文件。正在创建一个用于演示的虚拟文件。")
        dummy_data = [
            {'title': 'Bitcoin vs gold: Safe haven battle', 'category': '', 'summary': '', 'link': 'https://www.fxstreet.com/cryptocurrencies/news/bitcoin-vs-gold-safe-haven-battle-202010261034', 'publication_time': '2018-01-01 00:00:00', 'full_article': 'A new idea has been floating around in Safe Haven Trading...'},
            {'title': 'Intel Leaks Details On New CPU', 'category': '', 'summary': '', 'link': 'https://www.extremetech.com/gaming/261297-intel-leaks-details-new-desktop-core-i7-8809g-radeon-graphics', 'publication_time': '2018-01-02 08:15:16', 'full_article': 'Ever since Intel and AMD confirmed...'},
            {'title': 'US Fed Considers Rate Hike', 'category': 'Finance', 'summary': '', 'link': 'https://example.com/fed-rate-hike', 'publication_time': '2018-01-03 14:00:00', 'full_article': 'The US Federal Reserve is considering a new rate hike to combat inflation...'}
        ]
        os.makedirs(os.path.dirname(input_file_path) or '.', exist_ok=True)
        with open(input_file_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=4)
            
    # 4. 加载输入的新闻数据
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)[0:10]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：加载新闻数据失败: {e}")
        return

    # 5. [断点续运行] 加载已有的分析结果
    processed_links = set()
    all_results = []
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                # 确保文件不是空的
                content = f.read()
                if content:
                    all_results = json.loads(content)
                    # 创建一个包含已处理链接的集合，用于快速查找
                    for result in all_results:
                        if 'article_link' in result:
                            processed_links.add(result['article_link'])
            print(f"成功加载 {len(processed_links)} 条已有的分析结果。")
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告：无法读取已有的结果文件 '{output_file_path}'。将重新开始。错误: {e}")
            all_results = []

    # 6. 遍历并处理每篇文章
    print(f"\n共找到 {len(articles)} 篇文章。现在开始处理...")
    for i, article in enumerate(articles):
        title = article.get('title', '无标题')
        content = article.get('full_article', '')
        timestamp = article.get('publication_time', 'N/A')
        article_link = article.get('link', '')

        # 检查关键信息是否存在
        if not article_link or not content:
            print(f"跳过文章 '{title}'，原因：缺少链接或内容。")
            continue
        
        # [断点续运行] 检查文章是否已经被处理过
        if article_link in processed_links:
            print(f"({i+1}/{len(articles)}) 跳过已处理的文章: {title}")
            continue

        print(f"=============== 正在处理第 {i+1}/{len(articles)} 篇文章 ===============")
        
        # 7. 构建用户提示（User Prompt）
        user_prompt = f"Please analyze the following news article:\n\nTitle: {title}\n\nContent: {content}"

        # 8. 查询模型并获取结构化结果
        analysis_result = agent.get_response(user_prompt)

        # 9. 处理并保存结果
        if analysis_result and isinstance(analysis_result, dict):
            print(f"--- 成功分析: {title}")

            # 为方便追溯，将文章标题和链接也添加到结果中
            analysis_result['news_title'] = title
            analysis_result['article_link'] = article_link
            
            # 将新结果添加到总列表中
            all_results.append(analysis_result)

            # [边跑边存] 将更新后的完整列表写回文件
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)
                print(f"--- 结果已保存到 '{output_file_path}'")
                
                # 将刚刚处理过的链接加入集合，防止在同一次运行中重复处理
                processed_links.add(article_link)
            except IOError as e:
                print(f"!!! 严重错误：无法将结果保存到 '{output_file_path}'。错误: {e} !!!")
        else:
            print(f"\n--- 分析失败 ---")
            print(f"无法获取文章的有效结构化响应: '{title}'")
            print("-----------------------\n")

    print("\n\n======== 所有分析任务执行完毕 ========")


if __name__ == "__main__":
    run_analysis_pipeline()
