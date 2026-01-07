#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中医药疗效文本关键词挖掘与词云可视化
项目名称：中医药疗效文本关键词挖掘与词云可视化
作者：Python大作业
日期：2024年

技术介绍：
本项目基于HERB 2.0数据库，对中医药临床试验疗效描述文本进行关键词挖掘和可视化分析。
通过中文分词、停用词过滤、TF-IDF关键词提取等技术，挖掘中医药疗效文本中的关键信息。

主要技术：
1. 中文分词技术（Jieba）
2. TF-IDF关键词提取算法
3. 词频统计分析
4. 词云图可视化

输入：HERB_herb_info.csv, HERB_experiment_info.csv
输出：分词结果CSV、词云图PNG、关键词统计表CSV
"""

import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("中医药疗效文本关键词挖掘与词云可视化")
print("=" * 60)

# =============================================================================
# 1. 数据加载和预处理
# =============================================================================

def load_and_preprocess_data():
    """
    数据加载和预处理函数
    Returns: herb_info_df, experiment_df
    """
    print("\n📂 正在读取数据文件...")
    
    # 读取草药信息文件
    try:
        herb_info_df = pd.read_csv('HERB_herb_info.csv', sep='\t', encoding='utf-8')
        print(f"✅ 草药信息文件读取成功，共{len(herb_info_df)}行数据")
    except:
        herb_info_df = pd.read_csv('HERB_herb_info.csv', encoding='gbk', sep='\t')
        print(f"✅ 草药信息文件读取成功（GBK编码），共{len(herb_info_df)}行数据")
    
    # 读取实验信息文件
    try:
        experiment_df = pd.read_csv('HERB_experiment_info.csv', sep='\t', encoding='utf-8')
        print(f"✅ 实验信息文件读取成功，共{len(experiment_df)}行数据")
    except:
        experiment_df = pd.read_csv('HERB_experiment_info.csv', encoding='gbk', sep='\t')
        print(f"✅ 实验信息文件读取成功（GBK编码），共{len(experiment_df)}行数据")
    
    # 显示数据基本信息
    print("\n📋 草药信息数据字段：", list(herb_info_df.columns))
    print("📋 实验信息数据字段：", list(experiment_df.columns))
    
    return herb_info_df, experiment_df

# =============================================================================
# 2. 中文分词和停用词过滤
# =============================================================================

def create_stopwords_list():
    """
    创建中医药停用词列表
    Returns: 停用词列表
    """
    # 自定义中医药停用词列表
    chinese_stopwords = [
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这',
        '那', '这样', '因为', '所以', '但是', '然后', '如果', '可以', '应该', '可能', '已经', '现在', '时间', '时候', '我们', '他们', '这个', '那个', '这些', '那些',
        '进行', '通过', '对于', '关于', '根据', '按照', '由于', '为了', '除了', '除了...之外', '以及', '或者', '而且', '虽然', '尽管', '即使', '不管', '无论', '不仅', '而且'
    ]
    
    # 添加更多中医药相关停用词
    medical_stopwords = ['治疗', '药物', '中药', '草药', '患者', '疾病', '症状', '疗效', '效果', '使用', '应用', '方法', '研究', '实验', '临床', '数据', '结果', '分析', '统计', '显著',
                         '细胞', '组织', '基因', '表达', '蛋白', '分子', '机制', '通路', '模型', '动物', '小鼠', '大鼠', '人体', '体外', '体内', '剂量', '浓度', '时间', '周期']
    chinese_stopwords.extend(medical_stopwords)
    
    print(f"✅ 停用词列表已创建，共{len(chinese_stopwords)}个停用词")
    return chinese_stopwords

def chinese_segmentation(text, stopwords):
    """
    中文文本分词函数
    Args:
        text: 输入的中文文本
        stopwords: 停用词列表
    Returns:
        分词后的词语列表
    """
    if pd.isna(text):
        return []
    
    # 使用jieba进行精确模式分词
    words = jieba.cut(str(text), cut_all=False)
    
    # 过滤停用词和单字词
    filtered_words = [word for word in words 
                     if len(word) > 1 
                     and word not in stopwords 
                     and not re.match('^[0-9a-zA-Z]+$', word)]
    
    return filtered_words

# =============================================================================
# 3. TF-IDF关键词提取
# =============================================================================

def extract_tfidf_keywords(texts, max_features=1000):
    """
    TF-IDF关键词提取函数
    Args:
        texts: 文本列表
        max_features: 最大特征词数量
    Returns:
        特征词列表和TF-IDF矩阵
    """
    # 创建TF-IDF向量化器，根据文档数量调整参数
    if len(texts) < 10:
        # 文档数量较少时，调整参数
        tfidf_vectorizer = TfidfVectorizer(
            max_features=min(max_features, len(texts) * 5),
            min_df=1,
            max_df=1.0
        )
    else:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8
        )
    
    # 计算TF-IDF矩阵
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # 获取特征词
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    print(f"✅ TF-IDF矩阵计算完成，特征词数量：{len(feature_names)}")
    
    return feature_names, tfidf_matrix, tfidf_vectorizer

# =============================================================================
# 4. 词频统计和TOP20分析
# =============================================================================

def analyze_word_frequency(all_words):
    """
    词频统计分析函数
    Args:
        all_words: 所有分词结果列表
    Returns:
        词频统计结果
    """
    # 统计词频
    word_freq = Counter(all_words)
    
    # 获取TOP20高频词
    top20_words = word_freq.most_common(20)
    
    print("\n📊 TOP20高频关键词统计：")
    for i, (word, freq) in enumerate(top20_words, 1):
        print(f"{i:2d}. {word}: {freq}次")
    
    return word_freq, top20_words

# =============================================================================
# 5. 词云可视化
# =============================================================================

def generate_wordcloud(word_freq, title, filename):
    """
    生成词云图函数
    Args:
        word_freq: 词频统计结果
        title: 图表标题
        filename: 保存文件名
    """
    # 创建词云对象
    wordcloud = WordCloud(
        font_path='simhei.ttf',
        width=800,
        height=600,
        background_color='white',
        max_words=200
    ).generate_from_frequencies(word_freq)
    
    # 绘制词云图
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 词云图已保存：{filename}")
    plt.close()

# =============================================================================
# 6. 主函数
# =============================================================================

def main():
    """主函数"""
    
    # 1. 数据加载和预处理
    herb_info_df, experiment_df = load_and_preprocess_data()
    
    # 2. 创建停用词列表
    stopwords = create_stopwords_list()
    
    # 3. 中文分词处理
    print("\n📝 正在对草药功能描述进行分词...")
    
    # 对草药信息中的功能描述进行分词
    herb_info_df['segmented_function'] = herb_info_df['Function'].apply(
        lambda x: chinese_segmentation(x, stopwords) if pd.notna(x) else []
    )
    
    # 对适应症进行分词
    herb_info_df['segmented_indication'] = herb_info_df['Indication'].apply(
        lambda x: chinese_segmentation(x, stopwords) if pd.notna(x) else []
    )
    
    print(f"✅ 分词完成！共处理{len(herb_info_df)}条草药信息")
    
    # 4. 合并所有分词结果用于分析
    all_function_words = []
    for words in herb_info_df['segmented_function']:
        all_function_words.extend(words)
    
    all_indication_words = []
    for words in herb_info_df['segmented_indication']:
        all_indication_words.extend(words)
    
    all_words = all_function_words + all_indication_words
    
    print(f"总分词数量：{len(all_words)}个词语")
    
    # 5. 词频统计分析
    word_freq, top20_words = analyze_word_frequency(all_words)
    
    # 6. 生成总体词云图
    generate_wordcloud(word_freq, 
                      '中医药疗效文本关键词词云图', 
                      '中医药疗效关键词词云图.png')
    
    # 7. 准备TF-IDF分析的数据
    print("\n🔍 准备TF-IDF分析数据...")
    
    # 合并所有功能描述文本
    all_function_texts = []
    for idx, row in herb_info_df.iterrows():
        if pd.notna(row['Function']) and len(row['segmented_function']) > 0:
            text = ' '.join(row['segmented_function'])
            all_function_texts.append(text)
    
    print(f"可用于TF-IDF分析的文档数量：{len(all_function_texts)}")
    
    if len(all_function_texts) > 0:
        # 提取TF-IDF关键词
        feature_names, tfidf_matrix, tfidf_vectorizer = extract_tfidf_keywords(all_function_texts)
        
        # 计算每个词语的平均TF-IDF值
        tfidf_scores = tfidf_matrix.mean(axis=0).A1
        word_tfidf_df = pd.DataFrame({
            '词语': feature_names,
            'TF-IDF值': tfidf_scores
        }).sort_values('TF-IDF值', ascending=False)
        
        # 合并词频和TF-IDF值
        word_stats_df = pd.DataFrame(top20_words, columns=['词语', '词频'])
        word_stats_df = word_stats_df.merge(word_tfidf_df, on='词语', how='left')
        
        # 8. 保存分词结果
        # 检查列名，使用正确的列名
        if 'Herb_' in herb_info_df.columns:
            id_column = 'Herb_'
        else:
            id_column = herb_info_df.columns[0]  # 使用第一列作为ID
        
        segmented_data = herb_info_df[[id_column, 'Herb_cn_name', 'segmented_function', 'segmented_indication']].copy()
        segmented_data.to_csv('分词后的中医药数据.csv', index=False, encoding='utf-8-sig')
        print("✅ 分词后的结构化数据已保存：分词后的中医药数据.csv")
        
        # 9. 保存关键词统计表
        word_stats_df.to_csv('高频关键词TOP20统计表.csv', index=False, encoding='utf-8-sig')
        print("✅ 高频关键词统计表已保存：高频关键词TOP20统计表.csv")
        
        # 10. 显示TOP20关键词统计结果
        print("\n📊 高频关键词TOP20统计表（含词频和TF-IDF值）：")
        print(word_stats_df.head(20))
        
        # 11. 生成TOP20关键词条形图
        plt.figure(figsize=(14, 8))
        sns.barplot(data=word_stats_df.head(20), x='词频', y='词语', palette='viridis')
        plt.title('中医药疗效文本高频关键词TOP20', fontsize=16)
        plt.xlabel('词频', fontsize=12)
        plt.ylabel('关键词', fontsize=12)
        plt.tight_layout()
        plt.savefig('高频关键词TOP20条形图.png', dpi=300, bbox_inches='tight')
        print("✅ 高频关键词条形图已保存：高频关键词TOP20条形图.png")
        plt.close()
    
    print("\n" + "=" * 60)
    print("🎉 项目执行完成！")
    print("📁 输出文件清单：")
    print("   - 分词后的中医药数据.csv")
    print("   - 高频关键词TOP20统计表.csv") 
    print("   - 中医药疗效关键词词云图.png")
    print("   - 高频关键词TOP20条形图.png")
    print("=" * 60)

if __name__ == "__main__":
    main()