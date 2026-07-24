"""
RAG Triad 评估指标 - TruLens 2.7+ Metric API

三大核心指标：
  1. 答案相关性 (Question → Answer)
  2. 上下文相关性 (Question → Context)
  3. 有据性 (Context → Answer)
"""
from trulens.core.metric import Metric
from trulens.core.metric.selector import Selector
from trulens.providers.litellm import LiteLLM
import numpy as np


def build_rag_triad_metrics(provider: LiteLLM) -> list[Metric]:

    # 指标 1: 答案相关性 (Question → Answer)
    m_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="答案相关性",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    # 指标 2: 上下文相关性 (Question → Context)
    m_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="上下文相关性",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )

    # 指标 3: 有据性 (Context → Answer)
    m_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="有据性",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
        },
    )

    return [m_answer_relevance, m_context_relevance, m_groundedness]
