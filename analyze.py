import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from collections import Counter
from snownlp import SnowNLP
import requests
import argparse
from utils import get_data_loader, clean_text

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-8dc4ee20fc8a40a4906dc872534a53a7"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
def load_data():
    print("正在加载数据...")
    with open(COMMENTS_PATH, 'r', encoding='utf-8') as f:
        comments_data = json.load(f)
    
    with open(CONTENTS_PATH, 'r', encoding='utf-8') as f:
        contents_data = json.load(f)
    
    return comments_data, contents_data

# 统计有效数据数量
def count_valid_data(comments_data, contents_data):
    valid_comments = [comment for comment in comments_data if comment.get('content')]
    valid_contents = [content for content in contents_data if content.get('desc') or content.get('title')]
    
    print(f"评论数据数量: {len(valid_comments)}")
    print(f"内容数据数量: {len(valid_contents)}")
    
    return len(valid_comments), len(valid_contents)

# 提取评论有效内容
def extract_comments(comments_data):
    print("提取评论数据...")
    comments_df = pd.DataFrame(comments_data)
    # 选择需要的列
    if not comments_df.empty:
        selected_columns = ['comment_id', 'create_time', 'ip_location', 'note_id', 'content', 
                           'nickname', 'like_count', 'sub_comment_count']
        selected_columns = [col for col in selected_columns if col in comments_df.columns]
        comments_df = comments_df[selected_columns]
    
    # 保存到CSV
    comments_df.to_csv('comments_data.csv', index=False, encoding='utf-8-sig')
    print(f"评论数据已保存到 comments_data.csv")
    
    return comments_df

# 提取内容有效内容
def extract_contents(contents_data):
    print("提取内容数据...")
    contents_df = pd.DataFrame(contents_data)
    # 选择需要的列
    if not contents_df.empty:
        selected_columns = ['note_id', 'type', 'title', 'desc', 'time', 'nickname', 
                           'liked_count', 'comment_count', 'ip_location', 'tag_list']
        selected_columns = [col for col in selected_columns if col in contents_df.columns]
        contents_df = contents_df[selected_columns]
    
    # 保存到CSV
    contents_df.to_csv('contents_data.csv', index=False, encoding='utf-8-sig')
    print(f"内容数据已保存到 contents_data.csv")
    
    return contents_df

# 词频分析
def word_frequency_analysis(text_series):
    print("进行词频分析...")
    # 合并所有文本
    all_text = ' '.join(text_series.astype(str).tolist())
    
    # 使用jieba进行分词
    jieba.setLogLevel(20)  # 设置jieba的日志级别，避免输出过多信息
    words = jieba.cut(all_text)
    
    # 过滤停用词
    stopwords = ['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']
    filtered_words = [word for word in words if len(word) > 1 and word not in stopwords]
    
    # 统计词频
    word_counts = Counter(filtered_words)
    
    # 获取前30个高频词
    top_words = word_counts.most_common(30)
    
    # 绘制词频图
    words, counts = zip(*top_words)
    plt.figure(figsize=(12, 8))
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title('评论高频词统计')
    plt.tight_layout()
    plt.savefig('word_frequency.png')
    plt.close()
    
    print(f"词频分析图已保存到 word_frequency.png")
    
    return top_words, all_text

# 情感倾向分析
def sentiment_analysis(text_series):
    print("进行情感倾向分析...")
    sentiments = []
    for text in text_series:
        if pd.isna(text) or text == '':
            sentiments.append(0.5)  # 中性
            continue
        try:
            s = SnowNLP(str(text))
            sentiments.append(s.sentiments)
        except:
            sentiments.append(0.5)  # 处理异常情况
    
    # 创建情感分布图
    plt.figure(figsize=(10, 6))
    plt.hist(sentiments, bins=20)
    plt.title('情感倾向分布')
    plt.xlabel('情感得分 (0:消极 - 1:积极)')
    plt.ylabel('频次')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('sentiment_analysis.png')
    plt.close()
    
    # 计算情感统计数据
    positive = sum(1 for s in sentiments if s > 0.7)
    neutral = sum(1 for s in sentiments if 0.3 <= s <= 0.7)
    negative = sum(1 for s in sentiments if s < 0.3)
    total = len(sentiments)
    
    print(f"情感分析结果:")
    print(f"积极评论: {positive} ({positive/total*100:.2f}%)")
    print(f"中性评论: {neutral} ({neutral/total*100:.2f}%)")
    print(f"消极评论: {negative} ({negative/total*100:.2f}%)")
    print(f"情感分析图已保存到 sentiment_analysis.png")
    
    # 绘制饼图
    labels = ['积极', '中性', '消极']
    sizes = [positive, neutral, negative]
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('评论情感倾向比例')
    plt.savefig('sentiment_pie.png')
    plt.close()
    
    print(f"情感比例图已保存到 sentiment_pie.png")
    
    return sentiments

# 调用DeepSeek API进行三级编码
def deepseek_analysis(text):
    print("调用DeepSeek API进行三级编码分析...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    # 限制文本长度，避免超出API限制
    if len(text) > 8000:
        text = text[:8000]
    
    prompt = f"""
    请对以下网络文本数据进行三级编码分析，提取关键主题和模式：
    
    {text}
    
    请按照以下三级编码结构进行分析：
    1. 开放式编码：识别原始数据中的关键概念和类别
    2. 主轴编码：确定这些概念之间的关系和联系
    3. 选择性编码：整合所有类别，确定核心主题
    
    对于每个级别，请提供详细的编码结果和分析。
    """
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        analysis_result = result['choices'][0]['message']['content']
        
        # 保存分析结果到文件
        with open('deepseek_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(analysis_result)
        
        print(f"DeepSeek分析结果已保存到 deepseek_analysis.txt")
        return analysis_result
    except Exception as e:
        error_msg = f"API调用失败: {str(e)}"
        print(error_msg)
        return error_msg

# 主函数
def main():
    print("=== 小红书网络文本数据分析 ===")
    
    # 加载数据
    comments_data, contents_data = load_data()
    
    # 统计有效数据数量
    count_valid_data(comments_data, contents_data)
    
    # 提取有效内容
    comments_df = extract_comments(comments_data)
    contents_df = extract_contents(contents_data)
    
    # 词频分析
    top_words, all_text = word_frequency_analysis(comments_df['content'])
    
    # 情感分析
    sentiments = sentiment_analysis(comments_df['content'])
    
    # 准备DeepSeek分析的文本
    print("\n准备DeepSeek分析的文本样本...")
    analysis_text = "\n\n".join(comments_df['content'].dropna().astype(str).tolist()[:50])
    analysis_text += "\n\n" + "\n\n".join(contents_df['desc'].dropna().astype(str).tolist()[:20])
    
    # 保存样本文本
    with open('analysis_sample.txt', 'w', encoding='utf-8') as f:
        f.write(analysis_text)
    print(f"分析样本已保存到 analysis_sample.txt")
    
    # 询问用户是否进行DeepSeek分析
    choice = input("\n是否进行DeepSeek API分析? (y/n): ")
    if choice.lower() == 'y':
        deepseek_analysis(analysis_text)
    
    print("\n分析完成! 所有结果已保存到相应文件中。")
    print("您可以编辑 analysis_sample.txt 文件后重新运行程序进行自定义分析。")

if __name__ == "__main__":
    main()