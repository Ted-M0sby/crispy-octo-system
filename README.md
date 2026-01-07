# 中医药疗效文本关键词挖掘与词云可视化

## 项目简介

本项目基于中医药数据，通过文本挖掘技术提取关键词，并进行词云可视化展示。项目包含完整的数据处理、关键词提取、词频统计和可视化分析流程。

## 项目文件结构

```
中医药疗效文本关键词挖掘与词云可视化/
├── 数据文件/
│   ├── HERB_experiment_info.csv          # 中医药实验信息数据
│   ├── HERB_herb_info.csv                # 中医药药材信息数据
│   ├── 分词后的中医药数据.csv            # 经过分词处理的中医药数据
│   └── 高频关键词TOP20统计表.csv         # 高频关键词统计结果
├── 代码文件/
│   ├── 中医药疗效文本关键词挖掘与词云可视化.ipynb  # Jupyter Notebook分析代码
│   └── 中医药疗效文本关键词挖掘与词云可视化.py     # Python脚本分析代码
├── 可视化结果/
│   ├── 中医药疗效关键词词云图.png        # 生成的关键词词云图
│   └── 高频关键词TOP20条形图.png         # 高频关键词条形图
└── README.md                             # 项目说明文档
```

## 数据说明

### HERB_herb_info.csv
包含中医药药材的详细信息，字段包括：
- `Herb_`: 药材编号
- `Herb_pinyin_name`: 药材拼音名称
- `Herb_cn_name`: 药材中文名称
- `Herb_en_name`: 药材英文名称
- `Herb_latin_name`: 药材拉丁文名称
- `Properties`: 药性
- `Meridians`: 归经
- `UsePart`: 使用部位
- `Function`: 功能
- `Indication`: 主治
- `Toxicity`: 毒性
- `Clinical_manifestations`: 临床表现
- `Therapeutic_en_class`: 治疗类别（英文）
- `Therapeutic_cn_class`: 治疗类别（中文）
- `TCMID_id`: TCMID数据库ID
- `TCM_ID_id`: TCM数据库ID
- `SymMap_id`: SymMap数据库ID
- `TCMSP_id`: TCMSP数据库ID

### HERB_experiment_info.csv
包含中医药实验相关信息。

## 技术栈

- **编程语言**: Python
- **数据处理**: Pandas, NumPy
- **文本处理**: Jieba分词
- **可视化**: Matplotlib, WordCloud
- **开发环境**: Jupyter Notebook

## 主要功能

1. **数据预处理**: 清洗和整理中医药文本数据
2. **中文分词**: 使用Jieba对中医药文本进行分词处理
3. **关键词提取**: 基于词频统计提取高频关键词
4. **词云生成**: 创建直观的词云可视化图表
5. **统计分析**: 生成TOP20关键词统计表和条形图

## 使用方法

### 运行Jupyter Notebook
```bash
jupyter notebook "中医药疗效文本关键词挖掘与词云可视化.ipynb"
```

### 运行Python脚本
```bash
python "中医药疗效文本关键词挖掘与词云可视化.py"
```

## 输出结果

项目运行后将生成以下可视化结果：

1. **中医药疗效关键词词云图**: 直观展示高频关键词的分布情况
2. **高频关键词TOP20条形图**: 显示前20个高频关键词的词频统计
3. **高频关键词TOP20统计表**: 提供详细的关键词词频数据

## 项目特点

- **专业性**: 专注于中医药领域的文本挖掘
- **可视化**: 提供直观的词云和统计图表
- **完整性**: 包含从数据处理到结果展示的完整流程
- **可复用**: 代码结构清晰，易于修改和扩展

## 适用场景

- 中医药研究数据分析
- 中医药文本挖掘教学
- 中医药知识图谱构建
- 中医药疗效评估研究

## 注意事项

1. 确保已安装所需的Python依赖包
2. 数据文件路径需根据实际情况调整
3. 分词效果可能受中医药专业术语影响，可考虑使用中医药专业词典优化

## 依赖包

运行本项目需要安装以下Python包：

```bash
pip install pandas numpy jieba matplotlib wordcloud
```

##
联系邮箱naunnn@icloud。com
