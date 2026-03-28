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

import torch
import torch.nn as nn
import torch.nn.functional as F

class NewsImpactModel(nn.Module):
    """
    (版本 3: 接受 pred_time_tensor 用于预测)
    """
    
    # ... __init__, _get_params, gamma_pdf 方法与上一版相同 ...
    
    def __init__(self):
        super().__init__()
        
        # 初始化为 0.1 (softplus(0.1) ≈ 0.74), 梯度充足
        init_val = 1.0
        
        self.raw_k_short = nn.Parameter(torch.tensor(init_val))
        self.raw_theta_short = nn.Parameter(torch.tensor(init_val))
        self.raw_loc_short = nn.Parameter(torch.tensor(init_val))
    # ... (对 medium 和 long 也一样) ...
        # 2. Medium-term
        self.raw_k_medium = nn.Parameter(torch.tensor(init_val))
        self.raw_theta_medium = nn.Parameter(torch.tensor(init_val))
        self.raw_loc_medium = nn.Parameter(torch.tensor(init_val)*7.0)
        # 3. Long-term
        self.raw_k_long = nn.Parameter(torch.tensor(init_val))
        self.raw_theta_long = nn.Parameter(torch.tensor(init_val))
        self.raw_loc_long = nn.Parameter(torch.tensor(init_val)*15.0)

    def _get_params(self):
        """
        应用约束 (已修正：强制 k >= 1.0 + 1e-6)
        """
        # k (shape) > 1.0。这对解析地计算峰值 (mode) 至关重要。
        k_s = F.softplus(self.raw_k_short) + 1.0 + 1e-6
        k_m = F.softplus(self.raw_k_medium) + 1.0 + 1e-6
        k_l = F.softplus(self.raw_k_long) + 1.0 + 1e-6
        
        # theta (scale) > 0
        t_s = F.softplus(self.raw_theta_short) + 1e-6
        t_m = F.softplus(self.raw_theta_medium) + 1e-6
        t_l = F.softplus(self.raw_theta_long) + 1e-6
        
        # loc (location/offset) >= 1.0
        l_s = F.softplus(self.raw_loc_short) 
        l_m = F.softplus(self.raw_loc_medium)
        l_l = F.softplus(self.raw_loc_long) 
        
        return (k_s, t_s, l_s), (k_m, t_m, l_m), (k_l, t_l, l_l)

    def gamma_pdf(self, x, k, theta, loc):
        # x: 距离新闻日的天数 [0, 1, 2, ...]
        x_shifted = x - loc 
        valid_mask = (x_shifted > 0).float()
        dist = torch.distributions.Gamma(k, 1.0 / theta)
        log_pdf = dist.log_prob(torch.clamp(x_shifted, min=1e-9))
        pdf = log_pdf.exp()
        pdf = pdf * valid_mask 
        return pdf

    # -----------------------------------------------------------------
    # vvvv 这里是您请求的修改 vvvv
    # -----------------------------------------------------------------

    def forward(self, preprocessed_news, pred_time_tensor):
        """
        计算 'preprocessed_news' (历史新闻) 
        对 'pred_time_tensor' (未来时间) 的累积影响。
        
        参数:
        preprocessed_news (list): load_and_preprocess_news 的输出,
                                  每条新闻必须包含 'day_index' (历史发生日)。
        pred_time_tensor (tensor): [num_pred_days], 我们要计算信号的时间轴。
                                   (例如 [100, 101, 102, ...])
        """
        
        # 1. 我们的输出信号将与 pred_time_tensor 长度相同
        num_pred_days = pred_time_tensor.shape[0]
        num_news = len(preprocessed_news)
        
        if num_news == 0:
            return torch.zeros(num_pred_days)
            
        (k_s, t_s, l_s), (k_m, t_m, l_m), (k_l, t_l, l_l) = self._get_params()
        
        # 2. 影响曲线现在的维度是 [新闻条数, 预测天数]
        all_impact_curves = torch.zeros(num_news, num_pred_days)

        for i, news_item in enumerate(preprocessed_news):
            if 'day_index' not in news_item or news_item['day_index'] is None:
                continue 
            
            # 3. (关键变化) 
            #    计算 "预测的每一天" 距离 "新闻发生的历史日期" 有多少天
            
            # news_day_index 是历史日期, e.g., 50
            news_day_index = news_item['day_index'] 
            
            # pred_time_tensor 是我们要计算的未来日期, e.g., [100, 101, 102]
            # days_since_news 将是 [50, 51, 52]
            days_since_news = pred_time_tensor - news_day_index
            
            # 4. (关键变化) 
            #    causal_mask 确保我们只计算新闻发生后的影响。
            #    如果 pred_time_tensor 意外地包含了历史日期 (e.g., 40),
            #    days_since_news 会是负数 (e.g., -10), mask 会将其设为0。
            causal_mask = (days_since_news >= 0).float()
            
            # days_for_pdf 是用于输入 Gamma 函数的天数
            days_for_pdf = torch.clamp(days_since_news, min=0) 
            
            # --- 归一化逻辑 (与之前完全相同) ---

            # 1. 计算短期影响
            pct_s = news_item['pct_short']
            if pct_s != 0:
                raw_shape_short = self.gamma_pdf(days_for_pdf, k_s, t_s, l_s)
                peak_value_short = torch.max(raw_shape_short) + 1e-9
                normalized_shape_short = raw_shape_short / peak_value_short
                impact_short = pct_s * normalized_shape_short
            else:
                impact_short = torch.zeros_like(days_for_pdf)

            # 2. 计算中期影响
            pct_m = news_item['pct_medium']
            if pct_m != 0:
                raw_shape_medium = self.gamma_pdf(days_for_pdf, k_m, t_m, l_m)
                peak_value_medium = torch.max(raw_shape_medium) + 1e-9
                normalized_shape_medium = raw_shape_medium / peak_value_medium
                impact_medium = pct_m * normalized_shape_medium
            else:
                impact_medium = torch.zeros_like(days_for_pdf)

            # 3. 计算长期影响
            pct_l = news_item['pct_long']
            if pct_l != 0:
                raw_shape_long = self.gamma_pdf(days_for_pdf, k_l, t_l, l_l)
                peak_value_long = torch.max(raw_shape_long) + 1e-9
                normalized_shape_long = raw_shape_long / peak_value_long
                impact_long = pct_l * normalized_shape_long
            else:
                impact_long = torch.zeros_like(days_for_pdf)
            
            # --- 归一化逻辑结束 ---

            # 总影响 = 三种曲线叠加
            total_impact_curve = impact_short + impact_medium + impact_long
            
            # 乘以新闻的相关性
            total_impact_curve = total_impact_curve * news_item['relevance']

            # 确保影响只在新闻发生后计算
            total_impact_curve = total_impact_curve * causal_mask
            
            all_impact_curves[i, :] = total_impact_curve

        # 聚合：按天求所有新闻影响的 *总和*
        daily_impact_signal = torch.sum(all_impact_curves, dim=0)
        
        return daily_impact_signal


class FullPredictionModel0(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 2. 我们的新闻影响模型
        self.news_model = NewsImpactModel()
        
    def forward(self, base_prediction, preprocessed_news, timeline_tensor):
        
        # 1. 得到基础预测 (例如：价格)
        # (假设 base_features 形状为 [batch_size, num_features])
        # base_prediction 形状为 [batch_size, num_days]
        
        # 2. 得到新闻影响信号
        # news_signal 形状为 [num_days]
        news_signal = self.news_model(preprocessed_news, timeline_tensor)
        
        # 3. 改造原有的预测结果
        # 使用广播 (unsqueeze) 使 news_signal 变为 [1, num_days]
        # "改造"方式： P_final = P_base * (1 + signal)
        # 这假设 P_base 是价格，signal 是百分比变化
        print(news_signal.unsqueeze(0))
        final_prediction = base_prediction * (1.0 + news_signal.unsqueeze(0))
        
        # 或者，如果 P_base 是价格的 *变动*，可以用加法
        # final_prediction = base_prediction + news_signal.unsqueeze(0)
        
        return final_prediction

import numpy as np
import json
import pandas as pd
import torch.optim as optim
print("开始处理数据和模型...")

# --- 准备数据 ---
RELEVANCE_THRESHOLD = 0.90
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
    timeline = pd.date_range(start='2020-11-28', end='2020-12-31', freq='D')
    # timeline = pd.date_range(start='2020-11-28', end='2021-05-20', freq='D')
    num_days = len(timeline)

    # 为新闻数据添加 'day_index'
    timeline_df = pd.DataFrame({'date': timeline})
    news_df = pd.DataFrame(preprocessed_news)
    
    # preprocessed_news_with_index = []
    # if not news_df.empty:
    #     # 合并以找到每条新闻对应的日期索引
    #     merged = pd.merge_asof(news_df.sort_values('timestamp'), 
    #                            timeline_df.reset_index().rename(columns={'index':'day_index'}), 
    #                            left_on='timestamp', 
    #                            right_on='date',
    #                            direction='forward') # 使用 'nearest' 或 'forward'
    if not news_df.empty:
        start_date = timeline.min()
        end_date = timeline.max()
        
        # 只保留在时间轴范围内的新闻
        # 确保 news_df['timestamp'] 也是 datetime 对象
        news_df = news_df[
            (news_df['timestamp'] >= start_date) & 
            (news_df['timestamp'] <= end_date)
        ]
        print(f"过滤后，剩下 {len(news_df)} 条新闻在时间轴范围内。")

    preprocessed_news_with_index = []
    if not news_df.empty:
        # 合并以找到每条新闻对应的日期索引
        merged = pd.merge_asof(news_df.sort_values('timestamp'), 
                               timeline_df.reset_index().rename(columns={'index':'day_index'}), 
                               left_on='timestamp', 
                               right_on='date',
                               direction='backward') # <--- 建议也改为 'backward'

        # 转回 Pytorch 训练所需的列表格式
        preprocessed_news_with_index = merged.to_dict('records')

        # 过滤掉不在时间轴上的新闻 (如果 merge_asof 找到了 None)
        preprocessed_news_with_index = [
            n for n in preprocessed_news_with_index if n['day_index'] is not None and not pd.isna(n['day_index'])
        ]
    # print(preprocessed_news_with_index[0])
    
    # print(f"创建了 {num_days} 天的时间轴。")
    print(f"加载了 {len(preprocessed_news_with_index)} 条相关新闻 (threshold={RELEVANCE_THRESHOLD})。")
    # raise

    # 创建 Pytorch 格式的时间轴 (float tensor)
    timeline_tensor = torch.arange(34, dtype=torch.float32)


    BATCH_SIZE = 5
    model = FullPredictionModel0()
    loss_function = nn.MSELoss()
    # 优化器将自动找到 model.news_model 内部的9个参数
    optimizer = optim.Adam(model.parameters(), lr=1)

    # 模拟基础模型的输入
    pred_cali = np.load('/scratch/u5ci/liuhongyuan.u5ci/timellm/pred.npy')[0:34,0,0]
    true_cali = np.load('/scratch/u5ci/liuhongyuan.u5ci/timellm/true.npy')[0:34,0,0]
    pred_cali= torch.tensor(pred_cali,dtype=torch.float32)
    true_cali= torch.tensor(true_cali,dtype=torch.float32)
    

    print("\n--- 开始模拟测试 ---")

    for epoch in range(40):
        # 1. 前向传播
        model.train()
        optimizer.zero_grad()
        # print(preprocessed_news_with_index)
        predictions = model(pred_cali[0].repeat(34), preprocessed_news_with_index, timeline_tensor)
        # print('results:',predictions,pred_cali,true_cali)
        # 2. 计算损失
        loss = loss_function(predictions, true_cali)
        
        # 3. 反向传播
        loss.backward()
        
        # 4. 更新参数
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
    print(true_cali/pred_cali[0].repeat(34)-1 )
    print("\n--- 训练完成 ---")
    print("模型中可学习的Gamma参数 (原始值):")
    for name, param in model.news_model.named_parameters():
        print(f"{name}: {param.data.item():.4f}")

    print("\n模型中可学习的Gamma参数 (约束后):")
    (k_s, t_s, l_s), (k_m, t_m, l_m), (k_l, t_l, l_l) = model.news_model._get_params()
    print(f"Short : k={k_s.item():.4f}, theta={t_s.item():.4f}, loc={l_s.item():.4f}")
    print(f"Medium: k={k_m.item():.4f}, theta={t_m.item():.4f}, loc={l_m.item():.4f}")
    print(f"Long  : k={k_l.item():.4f}, theta={t_l.item():.4f}, loc={l_l.item():.4f}")

    print('\n--- 测试开始 ---\n')


    if json_string_from_file:
        preprocessed_news = load_and_preprocess_news(json_string_from_file, RELEVANCE_THRESHOLD)

        # 假设我们的时间轴从 2017-12-01 到 2024-02-01 (您可以根据需要调整)
        # 确保时间轴覆盖您的所有新闻日期
        # timeline = pd.date_range(start='2020-11-28', end='2020-12-31', freq='D')
        timeline = pd.date_range(start='2020-12-31', end='2021-05-20', freq='D')
        num_days = len(timeline)

        # 为新闻数据添加 'day_index'
        timeline_df = pd.DataFrame({'date': timeline})
        news_df = pd.DataFrame(preprocessed_news)
        
        # preprocessed_news_with_index = []
        # if not news_df.empty:
        #     # 合并以找到每条新闻对应的日期索引
        #     merged = pd.merge_asof(news_df.sort_values('timestamp'), 
        #                            timeline_df.reset_index().rename(columns={'index':'day_index'}), 
        #                            left_on='timestamp', 
        #                            right_on='date',
        #                            direction='forward') # 使用 'nearest' 或 'forward'
        if not news_df.empty:
            start_date = timeline.min()
            end_date = timeline.max()
            
            # 只保留在时间轴范围内的新闻
            # 确保 news_df['timestamp'] 也是 datetime 对象
            news_df = news_df[
                (news_df['timestamp'] >= start_date) & 
                (news_df['timestamp'] <= end_date)
            ]
            print(f"过滤后，剩下 {len(news_df)} 条新闻在时间轴范围内。")

        preprocessed_news_with_index = []
        if not news_df.empty:
            # 合并以找到每条新闻对应的日期索引
            merged = pd.merge_asof(news_df.sort_values('timestamp'), 
                                timeline_df.reset_index().rename(columns={'index':'day_index'}), 
                                left_on='timestamp', 
                                right_on='date',
                                direction='backward') # <--- 建议也改为 'backward'

            # 转回 Pytorch 训练所需的列表格式
            preprocessed_news_with_index = merged.to_dict('records')

            # 过滤掉不在时间轴上的新闻 (如果 merge_asof 找到了 None)
            preprocessed_news_with_index = [
                n for n in preprocessed_news_with_index if n['day_index'] is not None and not pd.isna(n['day_index'])
            ]
        # print(preprocessed_news_with_index[0])
        
        # print(f"创建了 {num_days} 天的时间轴。")
        print(f"加载了 {len(preprocessed_news_with_index)} 条相关新闻 (threshold={RELEVANCE_THRESHOLD})。")
        # raise

        # 创建 Pytorch 格式的时间轴 (float tensor)
        timeline_tensor = torch.arange(140, dtype=torch.float32)


        BATCH_SIZE = 140
        model = FullPredictionModel0()
        loss_function = nn.MSELoss()
        # 优化器将自动找到 model.news_model 内部的9个参数
        optimizer = optim.Adam(model.parameters(), lr=1)

        # 模拟基础模型的输入
        pred_cali = np.load('/scratch/u5ci/liuhongyuan.u5ci/timellm/pred.npy')[34:,0,0]
        true_cali = np.load('/scratch/u5ci/liuhongyuan.u5ci/timellm/true.npy')[34:,0,0]
        pred_cali= torch.tensor(pred_cali,dtype=torch.float32)
        true_cali= torch.tensor(true_cali,dtype=torch.float32)

        predictions = model(pred_cali[0].repeat(140), preprocessed_news_with_index, timeline_tensor)
        loss = loss_function(predictions, true_cali)
        

        print(loss.item(),loss_function(pred_cali, true_cali).item(),loss_function(pred_cali[0].repeat(140), true_cali).item())
        np.save('/scratch/u5ci/liuhongyuan.u5ci/timellm/time_agent/test_pred.npy',predictions.detach().numpy())
        print(true_cali/pred_cali[0].repeat(140)-1 )

    print("\n--- 测试完成 ---\n")




else:
    print("由于数据加载失败，训练未开始。")
