"""Worker-boundary tests for isolating retrieved memory from current user input."""

from src.agents.tools.worker_tools import _separate_inline_memory_context


def test_separates_legacy_supervisor_memory_suffix():
    message, memories = _separate_inline_memory_context(
        "我之前说的劳动争议是什么情况？[长期记忆] 用户在上海，老板拖欠三个月工资"
    )

    assert message == "我之前说的劳动争议是什么情况？"
    assert memories == ["用户在上海，老板拖欠三个月工资"]


def test_preserves_plain_user_message():
    message, memories = _separate_inline_memory_context("公司三个月没发工资")

    assert message == "公司三个月没发工资"
    assert memories == []
