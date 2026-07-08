"""
Jieba 自定义词库 —— 热加载 + Git 版本管理

核心认知：
- jieba 默认词库不认识内部术语："多AZ"→"多"+"AZ"，"优化顾问"→"优化"+"顾问"
- 自定义词库：纯文本文件，每行一个词
- 词库文件放 Git 仓库，修改后 commit + push，调 reload 接口拉最新版本
- 运行时热加载：调 jieba.load_userdict() 即可，30s 内全节点生效，不用重启

API 要点（面试用）：
- jieba.load_userdict(path) → 加载自定义词库
- jieba.add_word(word, freq, tag) → 逐条添加
- jieba.suggest_freq(segment, tune=True) → 调整分词概率
"""

import jieba

DICT_PATH = "data/jieba_dict.txt"


# --- 词库文件格式（data/jieba_dict.txt）---
# 每行一个词，格式：词 词频 词性（词频和词性可选）
DICT_FILE_EXAMPLE = """
优化顾问 100
region白名单 100
多AZ 100 nz
单AZ 100 nz
跨AZ 100 nz
多地域 100
SMS通知 100
OA监控告警 100
BSOD诊断 100 nz
蓝屏工具 100
"""


# --- 初始化加载（应用启动时）---
def init_jieba():
    """FastAPI 启动时加载自定义词库"""
    jieba.load_userdict(DICT_PATH)


# --- 运行时热加载（改词库后调一次，不用重启）---
def reload_jieba_dict():
    """
    热加载词库 —— 三步操作：

    1. 开发者修改 data/jieba_dict.txt，commit + push 到 Git 仓库
    2. 调 FastAPI 的 reload 接口：POST /admin/jieba/reload
    3. 接口从 Git 拉最新词库文件 + jieba.load_userdict()

    4 个节点各自调一次 reload，30s 内全部生效
    """
    # 实际流程：
    # 1. git pull origin main  # 或从配置中心拉取
    # 2. jieba.load_userdict(DICT_PATH)
    # 3. 验证：用已知问题 query 测试分词结果
    jieba.load_userdict(DICT_PATH)

    # 验证分词修复（Q31 案例）
    test_query = "多AZ部署的region白名单怎么配"
    tokens = list(jieba.cut(test_query))
    # 期望：['多AZ', '部署', '的', 'region白名单', '怎么', '配']
    # 修复前：['多', 'AZ', '部署', '的', 'region', '白名单', '怎么', '配']
    return {"status": "ok", "test_tokens": tokens}


# --- 逐条添加（灵活但无版本记录，不推荐）---
def add_word_hot(word: str, freq: int = 100, tag: str = ""):
    """逐条添加词 —— 灵活但缺少版本记录，不如维护词库文件"""
    jieba.add_word(word, freq, tag)


# --- 添加策略（Q29 + Q31）---
# 不只是加"当前这一个"，顺手加同系列的
# 比如发现"多AZ"被切错 → 加"多AZ"、"单AZ"、"跨AZ"、"多地域"
# 一次修复覆盖一类问题
def add_term_with_variants(term: str, variants: list[str]):
    """
    发现一个术语被切错，顺带加同系列变体
    add_term_with_variants("多AZ", ["单AZ", "跨AZ", "多地域", "多可用区"])
    """
    for word in [term] + variants:
        jieba.add_word(word, freq=100)
