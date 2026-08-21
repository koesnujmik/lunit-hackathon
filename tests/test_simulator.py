import asyncio
from unittest.mock import AsyncMock, Mock

from l2_baseline.simulator import next_patient_message


def test_patient_simulator_retries_502() -> None:
    failed = Mock(status_code=502)
    success = Mock(status_code=200)
    success.json.return_value = {"choices": [{"message": {"content": "질문"}}]}
    client = AsyncMock()
    client.post.side_effect = [failed, success]

    result = asyncio.run(next_patient_message(client, "https://example.test", []))
    assert result == "질문"
    assert client.post.await_count == 2
