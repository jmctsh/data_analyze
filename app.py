import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import jieba.analyse
import networkx as nx
from collections import Counter
from wordcloud import WordCloud
from snownlp import SnowNLP
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import matplotlib
import argparse
from utils import get_data_loader, clean_text

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-8dc4ee20fc8a40a4906dc872534a53a7"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 获取命令行参数
def get_platform_from_args():
    parser = argparse.ArgumentParser(description='社交媒体数据分析工具')
    parser.add_argument('--platform', type=str, default='xhs', 
                        choices=['xhs', 'dy'], 
                        help='指定要分析的平台: xhs (小红书) 或 dy (抖音)')
    args, _ = parser.parse_known_args()
    return args.platform

# 词频分析
def word_frequency_analysis(text_series):
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
    
    return top_words, all_text

# 生成词云
def generate_wordcloud(text):
    wordcloud = WordCloud(font_path='simhei.ttf', width=800, height=400, 
                         background_color='white', max_words=100).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存词云图
    wordcloud_path = 'wordcloud.png'
    plt.savefig(wordcloud_path)
    plt.close()
    
    return wordcloud_path

# 语义网络分析
def semantic_network_analysis(top_words, text):
    # 创建图
    G = nx.Graph()
    
    # 添加节点
    for word, count in top_words:
        G.add_node(word, weight=count)
    
    # 添加边 - 基于共现关系
    words_dict = dict(top_words)
    words_list = list(words_dict.keys())
    
    # 检查每对高频词是否在同一段文本中共现
    sentences = re.split(r'[。！？.!?]', text)
    for i, word1 in enumerate(words_list):
        for word2 in words_list[i+1:]:
            weight = 0
            for sentence in sentences:
                if word1 in sentence and word2 in sentence:
                    weight += 1
            if weight > 0:
                G.add_edge(word1, word2, weight=weight)
    
    # 绘制语义网络图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # 使用spring布局
    
    # 节点大小基于词频
    node_size = [G.nodes[node]['weight'] * 20 for node in G.nodes()]
    
    # 边的宽度基于共现频率
    edge_width = [G[u][v]['weight'] / 2 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='SimHei')
    
    plt.axis('off')
    plt.tight_layout()
    
    # 保存语义网络图
    network_path = 'semantic_network.png'
    plt.savefig(network_path)
    plt.close()
    
    return network_path

# 情感倾向分析
def sentiment_analysis(text_series):
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
    sns.histplot(sentiments, bins=20, kde=True)
    plt.title('情感倾向分布')
    plt.xlabel('情感得分 (0:消极 - 1:积极)')
    plt.ylabel('频次')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存情感分析图
    sentiment_path = 'sentiment_analysis.png'
    plt.savefig(sentiment_path)
    plt.close()
    
    # 计算情感统计数据
    positive = sum(1 for s in sentiments if s > 0.7)
    neutral = sum(1 for s in sentiments if 0.3 <= s <= 0.7)
    negative = sum(1 for s in sentiments if s < 0.3)
    total = len(sentiments)
    
    sentiment_stats = {
        'positive': positive,
        'positive_percent': positive / total * 100 if total > 0 else 0,
        'neutral': neutral,
        'neutral_percent': neutral / total * 100 if total > 0 else 0,
        'negative': negative,
        'negative_percent': negative / total * 100 if total > 0 else 0,
    }
    
    return sentiment_path, sentiment_stats, sentiments

# 调用DeepSeek API进行三级编码
def deepseek_analysis(text):
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
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"API调用失败: {str(e)}"

# Streamlit应用
def main():
    # 获取平台参数
    platform = get_platform_from_args()
    
    # 设置页面配置
    platform_name = "小红书" if platform == "xhs" else "抖音"
    st.set_page_config(page_title=f"{platform_name}数据分析", page_icon="📊", layout="wide")
    
    st.title(f"{platform_name}网络文本数据分析")
    
    # 平台选择器
    platform_options = {"xhs": "小红书", "dy": "抖音"}
    selected_platform = st.sidebar.selectbox(
        "选择分析平台",
        options=list(platform_options.keys()),
        format_func=lambda x: platform_options[x],
        index=list(platform_options.keys()).index(platform)
    )
    
    # 获取对应平台的数据加载器
    data_loader = get_data_loader(selected_platform)
    
    # 加载数据
    with st.spinner("正在加载数据..."):
        comments_data, contents_data = data_loader.load_data()
        # 数据清洗
        comments_data, contents_data = data_loader.clean_data(comments_data, contents_data)
    
    # 统计有效数据数量
    valid_comments = [comment for comment in comments_data if comment.get('content')]
    valid_contents = [content for content in contents_data if content.get('desc') or content.get('title')]
    valid_comments_count = len(valid_comments)
    valid_contents_count = len(valid_contents)
    
    # 显示数据统计
    st.header("1. 数据统计")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("评论数据数量", valid_comments_count)
    with col2:
        st.metric("内容数据数量", valid_contents_count)
    
    # 提取有效内容
    comments_df = data_loader.extract_comments(comments_data)
    contents_df = data_loader.extract_contents(contents_data)
    
    # 数据预览
    st.header("2. 数据预览")
    tab1, tab2 = st.tabs(["评论数据", "内容数据"])
    
    with tab1:
        st.subheader("评论数据")
        st.dataframe(comments_df.head(10))
        
        # 评论数据下载
        csv = comments_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载评论数据CSV",
            data=csv,
            file_name="comments_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("内容数据")
        st.dataframe(contents_df.head(10))
        
        # 内容数据下载
        csv = contents_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载内容数据CSV",
            data=csv,
            file_name="contents_data.csv",
            mime="text/csv"
        )
    
    # 舆情分析
    st.header("3. 舆情分析")
    
    # 词频分析
    st.subheader("3.1 词频分析")
    top_words, all_text = word_frequency_analysis(comments_df['content'])
    
    # 显示高频词
    df_top_words = pd.DataFrame(top_words, columns=['词语', '频次'])
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.dataframe(df_top_words)
    
    with col2:
        fig = px.bar(df_top_words, x='词语', y='频次', title='高频词统计')
        st.plotly_chart(fig, use_container_width=True)
    
    # 词云生成
    st.subheader("3.2 词云分析")
    wordcloud_path = generate_wordcloud(all_text)
    st.image(wordcloud_path, caption='评论词云')
    
    # 语义网络分析
    st.subheader("3.3 语义网络分析")
    network_path = semantic_network_analysis(top_words, all_text)
    st.image(network_path, caption='语义网络图')
    
    # 情感分析
    st.subheader("3.4 情感倾向分析")
    sentiment_path, sentiment_stats, sentiments = sentiment_analysis(comments_df['content'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(sentiment_path, caption='情感分布图')
    
    with col2:
        # 创建饼图
        labels = ['积极', '中性', '消极']
        values = [sentiment_stats['positive_percent'], 
                 sentiment_stats['neutral_percent'], 
                 sentiment_stats['negative_percent']]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title_text='情感倾向比例')
        st.plotly_chart(fig, use_container_width=True)
    
    # DeepSeek API分析
    st.header("4. DeepSeek文本分析与三级编码")
    
    # 合并评论和内容文本用于分析
    analysis_text = "\n\n".join(comments_df['content'].dropna().astype(str).tolist()[:50])
    analysis_text += "\n\n" + "\n\n".join(contents_df['desc'].dropna().astype(str).tolist()[:20])
    
    # 文本编辑区
    edited_text = st.text_area("编辑用于分析的文本", analysis_text, height=200)
    
    if st.button("进行DeepSeek分析"):
        with st.spinner("正在调用DeepSeek API进行分析..."):
            analysis_result = deepseek_analysis(edited_text)
            st.markdown("### 三级编码分析结果")
            st.markdown(analysis_result)
            
            # 提供复制功能
            st.text_area("复制分析结果", analysis_result, height=300)
    
    # 结果导出
    st.header("5. 结果导出")
    st.markdown("""
    您可以通过以下方式导出分析结果：
    1. 使用上方的下载按钮下载原始数据
    2. 右键点击图表保存图像
    3. 复制文本分析结果粘贴到您的论文中
    """)

if __name__ == "__main__":
    main()