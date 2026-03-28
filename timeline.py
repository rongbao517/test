import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import sys # 导入 sys 模块以便在需要时退出

# ---------------------------------------------------------------------------
# 1. 数据预处理
# ---------------------------------------------------------------------------

# 注意：原始的硬编码 json_data 字符串已被移除。
# 数据将从第 4 节中的文件路径加载。

def parse_impact_string(s):
    """将 '+0.10', '-0.05', '±0.05', '0.00' 转换为浮点数"""
    s = s.strip().replace('+', '')
    if '±' in s:
        # '±0.05' 解释为 0.0
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0

def load_and_preprocess_news(json_string, relevance_threshold=0.5):
    """
    读取JSON字符串，按日期排序，过滤，并解析影响。
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        print("请检查JSON文件格式是否正确。")
        return [] # 返回空列表
        
    processed = []
    for item in data:
        # 确保关键字段存在
        if 'relevance_to_bitcoin_price' not in item or \
           'publication_time' not in item or \
           'impact_analysis' not in item:
            print(f"警告: 跳过不完整的条目: {item.get('news_title', 'N/A')}")
            continue

        if item['relevance_to_bitcoin_price'] <= relevance_threshold:
            continue
            
        # 解析日期
        try:
            # 尝试解析标准格式
            timestamp = pd.to_datetime(item['publication_time'])
        except Exception:
            # 捕获像 'December 8-9, 2017' 这样的非标准日期
            print(f"警告: 跳过无法解析的日期: {item['publication_time']}")
            continue
            
        impact = item['impact_analysis']
        
        # 确保 impact 字段存在
        if 'short_term_impact_5_days' not in impact or \
           'medium_term_impact_15_days' not in impact or \
           'long_term_impact_after_15_days' not in impact:
            print(f"警告: 跳过缺少 impact_analysis 子字段的条目: {item.get('news_title', 'N/A')}")
            continue

        processed.append({
            'timestamp': timestamp,
            'title': item.get('news_title', 'N/A'), # 使用 .get 增加鲁棒性
            'relevance': item['relevance_to_bitcoin_price'],
            'pct_short': parse_impact_string(impact['short_term_impact_5_days']['percentage_change']),
            'pct_medium': parse_impact_string(impact['medium_term_impact_15_days']['percentage_change']),
            'pct_long': parse_impact_string(impact['long_term_impact_after_15_days']['percentage_change']),
        })
        
    # 按时间顺序排序
    processed.sort(key=lambda x: x['timestamp'])
    return processed

# ---------------------------------------------------------------------------
# 2. 定义 Gamma 分布影响模型
# ---------------------------------------------------------------------------

class NewsImpactModel(nn.Module):
    """
    实现可学习的9参数Gamma衰减模型
    """
    def __init__(self):
        super().__init__()
        
        # 我们将学习参数的 "原始" 版本
        # 然后使用 softplus 确保它们为正 (k > 0, theta > 0)
        # loc >= 1.0 (确保影响在新闻当天之后)

        # 1. Short-term (3 params)
        # (k, theta, loc) 或 (shape, scale, location)
        self.raw_k_short = nn.Parameter(torch.randn(1))
        self.raw_theta_short = nn.Parameter(torch.randn(1))
        self.raw_loc_short = nn.Parameter(torch.randn(1)) # loc >= 1

        # 2. Medium-term (3 params)
        self.raw_k_medium = nn.Parameter(torch.randn(1))
        self.raw_theta_medium = nn.Parameter(torch.randn(1))
        self.raw_loc_medium = nn.Parameter(torch.randn(1)) # loc >= 1

        # 3. Long-term (3 params)
        self.raw_k_long = nn.Parameter(torch.randn(1))
        self.raw_theta_long = nn.Parameter(torch.randn(1))
        self.raw_loc_long = nn.Parameter(torch.randn(1)) # loc >= 1

    def _get_params(self):
        """应用约束，确保参数有效"""
        # k (shape) > 0. 添加 1e-6 避免为0
        k_s = F.softplus(self.raw_k_short) + 1e-6
        k_m = F.softplus(self.raw_k_medium) + 1e-6
        k_l = F.softplus(self.raw_k_long) + 1e-6
        
        # theta (scale) > 0
        t_s = 7*F.softplus(self.raw_theta_short) + 1e-6
        t_m = 7*F.softplus(self.raw_theta_medium) + 1e-6
        t_l = 7*F.softplus(self.raw_theta_long) + 1e-6
        
        # loc (location/offset) >= 1.0 
        # 确保影响最早在新闻发布后的第1天开始
        l_s = 14*F.softplus(self.raw_loc_short) + 1.0
        l_m = 14*F.softplus(self.raw_loc_medium) + 1.0
        l_l = 14*F.softplus(self.raw_loc_long) + 1.0
        
        return (k_s, t_s, l_s), (k_m, t_m, l_m), (k_l, t_l, l_l)

    def gamma_pdf(self, x, k, theta, loc):
        """
        计算三参数 Gamma PDF(x | k, theta, loc)
        x: (tensor) 距离新闻日的天数
        k: (tensor) shape
        theta: (tensor) scale
        loc: (tensor) location (偏移量)
        """
        # x_shifted 是 Gamma 分布的输入 (必须 > 0)
        x_shifted = x + loc
        
        # Gamma(k, theta) = Gamma(concentration=k, rate=1/theta)
        dist = torch.distributions.Gamma(k, 1.0 / theta)
        
        # log_prob 在 x_shifted <= 0 时为 -inf
        log_pdf = dist.log_prob(x_shifted)
        
        # exp(-inf) = 0.
        pdf = log_pdf.exp()
        
        # 显式地将 x_shifted <= 0 的 PDF 设为 0
        pdf = torch.where(x_shifted > 0, pdf, 0.0)
        return pdf

    def forward(self, preprocessed_news, timeline_tensor):
        """
        preprocessed_news: load_and_preprocess_news 的输出
        timeline_tensor: [num_days], 包含时间轴上每一天的天数 (例如 float 形式的 OADate)
                         这里我们简化为自0开始的整数天：[0, 1, 2, ..., T]
        """
        num_days = timeline_tensor.shape[0]
        num_news = len(preprocessed_news)
        
        if num_news == 0:
            return torch.zeros(num_days)
            
        # 获取9个可学习的参数
        (k_s, t_s, l_s), (k_m, t_m, l_m), (k_l, t_l, l_l) = self._get_params()
        
        # [num_news, num_days]
        # all_impact_curves[i, j] = 第 i 条新闻对第 j 天的影响
        all_impact_curves = torch.zeros(num_news, num_days)

        for i, news_item in enumerate(preprocessed_news):
            # 找到新闻在时间轴上的日期索引
            # (这是一个简化。在实际中, timeline_tensor 应该与 news_item['timestamp'] 对齐)
            # 为了演示，我们假设 timeline_tensor[news_day_index] 对应 news_item['timestamp']
            
            # 这里我们使用一个更通用的方法：
            # 假设 timeline_tensor 是 [0, 1, 2, ..., T]
            # news_item['day_index'] 是新闻发生在哪一天
            
            if 'day_index' not in news_item or news_item['day_index'] is None:
                continue # 如果新闻不在时间轴内，跳过
                
            news_day_index = news_item['day_index'] 
            
            # days_since_news[j] = j - news_day_index
            # [num_days]
            days_since_news = timeline_tensor - news_day_index
            
            causal_mask = (days_since_news >= 0).float()
            days_for_pdf = torch.clamp(days_since_news, min=0)
            
            # 1. 计算短期影响曲线
            pct_s = news_item['pct_short']
            pdf_short = self.gamma_pdf(days_for_pdf, k_s, t_s, l_s)
            impact_short = pct_s * pdf_short
            
            # 2. 计算中期影响曲线
            pct_m = news_item['pct_medium']
            pdf_medium = self.gamma_pdf(days_for_pdf, k_m, t_m, l_m)
            impact_medium = pct_m * pdf_medium
            
            # 3. 计算长期影响曲线
            pct_l = news_item['pct_long']
            pdf_long = self.gamma_pdf(days_for_pdf, k_l, t_l, l_l)
            impact_long = pct_l * pdf_long
            
            # 总影响 = 三种曲线叠加
            total_impact_curve = impact_short + impact_medium + impact_long
            
            # 乘以新闻的相关性
            total_impact_curve = total_impact_curve * news_item['relevance']

            total_impact_curve = total_impact_curve * causal_mask
            
            all_impact_curves[i, :] = total_impact_curve

        # 聚合：按天求所有新闻影响的平均值
        # [num_days]
        daily_impact_signal = torch.mean(all_impact_curves, dim=0)
        
        return daily_impact_signal


# ---------------------------------------------------------------------------
# 3. 定义完整的预测模型
# ---------------------------------------------------------------------------

class FullPredictionModel(nn.Module):
    def __init__(self, num_features, num_days):
        super().__init__()
        
        # 1. 您的 "原有的时序预测结果" (这里用一个简单的线性层模拟)
        # 它可以是 LSTM, GRU, ARIMA 等的输出
        # 假设它接收一些特征 (num_features) 并预测每天的价格
        self.base_model = nn.Linear(num_features, num_days)
        
        # 2. 我们的新闻影响模型
        self.news_model = NewsImpactModel()
        
    def forward(self, base_features, preprocessed_news, timeline_tensor):
        
        # 1. 得到基础预测 (例如：价格)
        # (假设 base_features 形状为 [batch_size, num_features])
        # base_prediction 形状为 [batch_size, num_days]
        base_prediction = self.base_model(base_features)
        
        # 2. 得到新闻影响信号
        # news_signal 形状为 [num_days]
        news_signal = self.news_model(preprocessed_news, timeline_tensor)
        
        # 3. 改造原有的预测结果
        # 使用广播 (unsqueeze) 使 news_signal 变为 [1, num_days]
        # "改造"方式： P_final = P_base * (1 + signal)
        # 这假设 P_base 是价格，signal 是百分比变化
        final_prediction = base_prediction * (1.0 + news_signal.unsqueeze(0))
        
        # 或者，如果 P_base 是价格的 *变动*，可以用加法
        # final_prediction = base_prediction + news_signal.unsqueeze(0)
        
        return final_prediction

# ---------------------------------------------------------------------------
# 4. 训练循环 (演示)
# ---------------------------------------------------------------------------

print("开始处理数据和模型...")

# --- 准备数据 ---
RELEVANCE_THRESHOLD = 0.5
# *** 修改：从文件读取 ***
FILE_PATH = "/scratch/u5ci/liuhongyuan.u5ci/timellm/time_agent/bitcoin_news_analysis_results_full_processed.json"

json_string_from_file = ""
try:
    # 指定 encoding='utf-8' 是一个好习惯
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        json_string_from_file = f.read()
    print(f"成功从 {FILE_PATH} 读取数据。")
except FileNotFoundError:
    print(f"错误: 文件未找到 {FILE_PATH}")
    print("请检查文件路径是否正确。")
    # sys.exit(1) # 在实际应用中，这里应该停止执行
except Exception as e:
    print(f"读取文件时发生未知错误: {e}")
    # sys.exit(1)

# 只有在成功读取文件后才继续
if json_string_from_file:
    preprocessed_news = load_and_preprocess_news(json_string_from_file, RELEVANCE_THRESHOLD)

    # 假设我们的时间轴从 2017-12-01 到 2024-02-01 (您可以根据需要调整)
    # 确保时间轴覆盖您的所有新闻日期
    timeline = pd.date_range(start='2018-01-01', end='2022-04-22', freq='D')
    num_days = len(timeline)

    # 为新闻数据添加 'day_index'
    timeline_df = pd.DataFrame({'date': timeline})
    news_df = pd.DataFrame(preprocessed_news)
    
    preprocessed_news_with_index = []
    if not news_df.empty:
        # 合并以找到每条新闻对应的日期索引
        merged = pd.merge_asof(news_df.sort_values('timestamp'), 
                               timeline_df.reset_index().rename(columns={'index':'day_index'}), 
                               left_on='timestamp', 
                               right_on='date',
                               direction='forward') # 使用 'nearest' 或 'forward'

        # 转回 Pytorch 训练所需的列表格式
        preprocessed_news_with_index = merged.to_dict('records')

        # 过滤掉不在时间轴上的新闻 (如果 merge_asof 找到了 None)
        preprocessed_news_with_index = [
            n for n in preprocessed_news_with_index if n['day_index'] is not None and not pd.isna(n['day_index'])
        ]
    
    print(f"创建了 {num_days} 天的时间轴。")
    print(f"加载了 {len(preprocessed_news_with_index)} 条相关新闻 (threshold={RELEVANCE_THRESHOLD})。")

    # 创建 Pytorch 格式的时间轴 (float tensor)
    timeline_tensor = torch.arange(num_days, dtype=torch.float32)

    # --- 模拟输入和标签 ---
    NUM_FEATURES = 10 # 基础模型的输入特征数
    BATCH_SIZE = 1 # 假设我们一次预测整个时间序列

    # 模拟基础模型的输入
    dummy_features = torch.randn(BATCH_SIZE, NUM_FEATURES)

    # 模拟 "真实" 价格标签
    dummy_labels = torch.randn(BATCH_SIZE, num_days) * 10 + 50000 # 模拟价格

    # --- 初始化模型、损失和优化器 ---
    model = FullPredictionModel(num_features=NUM_FEATURES, num_days=num_days)
    loss_function = nn.MSELoss()
    # 优化器将自动找到 model.news_model 内部的9个参数
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\n--- 开始模拟训练 ---")

    for epoch in range(50):
        # 1. 前向传播
        model.train()
        optimizer.zero_grad()
        print(preprocessed_news_with_index)
        predictions = model(dummy_features, preprocessed_news_with_index, timeline_tensor)
        
        # 2. 计算损失
        loss = loss_function(predictions, dummy_labels)
        
        # 3. 反向传播
        loss.backward()
        
        # 4. 更新参数
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

    print("\n--- 训练完成 ---")
    print("模型中可学习的Gamma参数 (原始值):")
    for name, param in model.news_model.named_parameters():
        print(f"{name}: {param.data.item():.4f}")

    print("\n模型中可学习的Gamma参数 (约束后):")
    (k_s, t_s, l_s), (k_m, t_m, l_m), (k_l, t_l, l_l) = model.news_model._get_params()
    print(f"Short : k={k_s.item():.4f}, theta={t_s.item():.4f}, loc={l_s.item():.4f}")
    print(f"Medium: k={k_m.item():.4f}, theta={t_m.item():.4f}, loc={l_m.item():.4f}")
    print(f"Long  : k={k_l.item():.4f}, theta={t_l.item():.4f}, loc={l_l.item():.4f}")

else:
    print("由于数据加载失败，训练未开始。")
