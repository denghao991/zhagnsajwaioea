"""
Query 改写 —— 两层串行：映射表 + LLM

核心认知：
- 第一层：动态映射表（纯字符串替换，<1ms）—— 内部术语 → 标准术语
- 第二层：LLM 改写（约 100ms）—— 去口语化，对齐文档风格
- 串行设计：先替换再改写 → 保证专有名词不被 LLM 误改写
- 映射表存在配置中心，运行时热加载；维护来源是线上 bad case 复盘

API 要点（面试用）：
- 映射表：dict 结构，存配置中心，reload 接口实时生效
- LLM 改写：短 prompt，输入输出各 ~50 token，轻量调用
"""

import re


# --- 第一层：动态映射表（纯字符串替换，<1ms）---
# 存配置中心，运行时热加载，不用重启
# 维护来源：线上 bad case 复盘 —— 用户说"那个监控"，系统不知道是"OA监控告警"
TERM_MAPPING = {
    "OA":    "优化顾问",
    "那个监控": "OA监控告警",
    "蓝屏工具": "BSOD诊断",
    "AZ":    "可用区",
    "多AZ":  "多可用区",   # 注意："多AZ"也在 jieba 词库里，两边都要维护
}


def reload_mapping():
    """热加载映射表配置，调一次即生效"""
    # 实际从配置中心拉取: new_mapping = config_center.get("term_mapping")
    # TERM_MAPPING.update(new_mapping)
    pass


def mapping_replace(query: str) -> str:
    """
    第一层：映射表替换
    "OA监控告警的那个监控怎么配" → "优化顾问监控告警的OA监控告警怎么配"
    """
    result = query
    for slang, formal in TERM_MAPPING.items():
        result = result.replace(slang, formal)
    return result


# --- 第二层：LLM 改写（去口语化，~100ms）---
REWRITE_PROMPT = """你是一个查询改写器。将用户的口语化问题改写为正式的技术文档查询。
规则：
1. 保留所有数字和否定词，不要修改
2. 保留所有专有名词和缩写
3. 只做口语→正式的转换，不添加用户没问的内容

示例：
输入："手机收不到短信怎么办"
输出："SMS通知渠道未收到告警"

输入：{query}
输出："""


def llm_rewrite(query: str) -> str:
    """
    第二层：LLM 口语化去除
    "OA监控告警的OA监控告警怎么配" → "OA监控告警配置方法"

    Prompt 极短（~50 token 入、~50 token 出），一个轻量 LLM 调用，约 100ms
    """
    # 实际走内部 LLM API
    # response = llm_client.generate(
    #     model="deepseek-v4-flash",
    #     prompt=REWRITE_PROMPT.format(query=query),
    #     max_tokens=100,
    #     temperature=0,  # 确定性输出
    # )
    # return response.text.strip()
    pass


def rewrite_pipeline(query: str) -> str:
    """
    两层串行改写入口
    总耗时：<1ms（映射表）+ ~100ms（LLM）= ~100ms
    """
    # Step 1: 映射表替换（先替换，保证专有名词不会被 LLM 误改写）
    replaced = mapping_replace(query)
    # Step 2: LLM 去口语化
    rewritten = llm_rewrite(replaced)
    return rewritten
