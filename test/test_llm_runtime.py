import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.agents.legal_guide.llm_runtime import ainvoke_bounded


def test_zero_timeout_invokes_followup_without_wait_for_deadline():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    with patch("src.agents.legal_guide.llm_runtime.asyncio.wait_for") as wait_for:
        result = asyncio.run(ainvoke_bounded(
            llm,
            [],
            timeout=0,
            stage="followup_batch_planner",
        ))

    wait_for.assert_not_called()
    assert result.content == "ok"
