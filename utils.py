import re
import json
import os
import pandas as pd

# 数据清洗函数
def clean_text(text):
    """
    清洗文本数据，去除表情符号和@用户信息
    
    Args:
        text (str): 需要清洗的文本
        
    Returns:
        str: 清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 去除[表情符号]，如[偷笑]，[爱心R]等
    text = re.sub(r'\[.*?\]', '', text)
    
    # 去除@用户信息，格式为@用户名 (注意空格)
    text = re.sub(r'@[^\s]+\s', '', text)
    
    return text.strip()

# 平台数据加载器基类
class PlatformDataLoader:
    def __init__(self, platform_dir):
        self.platform_dir = platform_dir
        self.comments_path = os.path.join(platform_dir, 'search_comments_2025-04-12.json')
        self.contents_path = os.path.join(platform_dir, 'search_contents_2025-04-12.json')
    
    def load_data(self):
        """
        加载评论和内容数据
        
        Returns:
            tuple: (comments_data, contents_data)
        """
        try:
            with open(self.comments_path, 'r', encoding='utf-8') as f:
                comments_data = json.load(f)
            
            with open(self.contents_path, 'r', encoding='utf-8') as f:
                contents_data = json.load(f)
                
            return comments_data, contents_data
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            return [], []
    
    def clean_data(self, comments_data, contents_data):
        """
        清洗评论和内容数据
        
        Args:
            comments_data (list): 评论数据列表
            contents_data (list): 内容数据列表
            
        Returns:
            tuple: (cleaned_comments_data, cleaned_contents_data)
        """
        # 子类需要实现此方法
        raise NotImplementedError("子类需要实现clean_data方法")

# 小红书数据加载器
class XhsDataLoader(PlatformDataLoader):
    def __init__(self):
        super().__init__('xhs')
    
    def clean_data(self, comments_data, contents_data):
        # 清洗评论数据
        for comment in comments_data:
            if 'content' in comment and comment['content']:
                comment['content'] = clean_text(comment['content'])
        
        # 清洗内容数据
        for content in contents_data:
            if 'title' in content and content['title']:
                content['title'] = clean_text(content['title'])
            if 'desc' in content and content['desc']:
                content['desc'] = clean_text(content['desc'])
        
        return comments_data, contents_data
    
    def extract_comments(self, comments_data):
        """
        提取小红书评论有效内容
        
        Args:
            comments_data (list): 评论数据列表
            
        Returns:
            DataFrame: 处理后的评论数据框
        """
        comments_df = pd.DataFrame(comments_data)
        # 选择需要的列
        if not comments_df.empty:
            selected_columns = ['comment_id', 'create_time', 'ip_location', 'note_id', 'content', 
                              'nickname', 'like_count', 'sub_comment_count']
            selected_columns = [col for col in selected_columns if col in comments_df.columns]
            comments_df = comments_df[selected_columns]
        
        return comments_df
    
    def extract_contents(self, contents_data):
        """
        提取小红书内容有效内容
        
        Args:
            contents_data (list): 内容数据列表
            
        Returns:
            DataFrame: 处理后的内容数据框
        """
        contents_df = pd.DataFrame(contents_data)
        # 选择需要的列
        if not contents_df.empty:
            selected_columns = ['note_id', 'type', 'title', 'desc', 'time', 'nickname', 
                              'liked_count', 'comment_count', 'ip_location', 'tag_list']
            selected_columns = [col for col in selected_columns if col in contents_df.columns]
            contents_df = contents_df[selected_columns]
        
        return contents_df

# 抖音数据加载器
class DouyinDataLoader(PlatformDataLoader):
    def __init__(self):
        super().__init__('dy')
    
    def clean_data(self, comments_data, contents_data):
        # 清洗评论数据
        for comment in comments_data:
            if 'content' in comment and comment['content']:
                comment['content'] = clean_text(comment['content'])
        
        # 清洗内容数据
        for content in contents_data:
            if 'title' in content and content['title']:
                content['title'] = clean_text(content['title'])
            if 'desc' in content and content['desc']:
                content['desc'] = clean_text(content['desc'])
        
        return comments_data, contents_data
    
    def extract_comments(self, comments_data):
        """
        提取抖音评论有效内容
        
        Args:
            comments_data (list): 评论数据列表
            
        Returns:
            DataFrame: 处理后的评论数据框
        """
        comments_df = pd.DataFrame(comments_data)
        # 选择需要的列
        if not comments_df.empty:
            selected_columns = ['comment_id', 'create_time', 'ip_location', 'aweme_id', 'content', 
                              'nickname', 'like_count', 'sub_comment_count']
            selected_columns = [col for col in selected_columns if col in comments_df.columns]
            comments_df = comments_df[selected_columns]
        
        return comments_df
    
    def extract_contents(self, contents_data):
        """
        提取抖音内容有效内容
        
        Args:
            contents_data (list): 内容数据列表
            
        Returns:
            DataFrame: 处理后的内容数据框
        """
        contents_df = pd.DataFrame(contents_data)
        # 选择需要的列
        if not contents_df.empty:
            selected_columns = ['aweme_id', 'aweme_type', 'title', 'desc', 'create_time', 'nickname', 
                              'liked_count', 'comment_count', 'ip_location', 'share_count']
            selected_columns = [col for col in selected_columns if col in contents_df.columns]
            contents_df = contents_df[selected_columns]
        
        return contents_df

# 数据加载器工厂
def get_data_loader(platform):
    """
    获取指定平台的数据加载器
    
    Args:
        platform (str): 平台名称，'xhs'或'dy'
        
    Returns:
        PlatformDataLoader: 对应平台的数据加载器实例
    """
    if platform.lower() == 'xhs':
        return XhsDataLoader()
    elif platform.lower() == 'dy':
        return DouyinDataLoader()
    else:
        raise ValueError(f"不支持的平台: {platform}，目前支持的平台有: xhs, dy")