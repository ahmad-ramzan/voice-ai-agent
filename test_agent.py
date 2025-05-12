import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from agent import entrypoint
from livekit.agents import JobContext, JobProcess, AutoSubscribe
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector

# Test if the room is created and the agent says the greeting message
@pytest.mark.asyncio
async def test_agent_start_and_greet():
    # Mock the JobContext and its methods
    mock_job_context = MagicMock(JobContext)
    mock_job_context.room.name = "test-room"
    mock_participant = MagicMock()
    mock_participant.identity = "participant-123"
    mock_job_context.wait_for_participant = AsyncMock(return_value=mock_participant)
    mock_job_context.connect = AsyncMock()

    # Mock the VoicePipelineAgent and its methods
    with patch("livekit.agents.pipeline.VoicePipelineAgent") as MockAgent:
        mock_agent = MagicMock(MockAgent)
        mock_agent.on = AsyncMock()
        mock_agent.say = AsyncMock()
        mock_agent.start = AsyncMock()

        # Mocking the plugins
        with patch("deepgram.STT", return_value=MagicMock()), \
             patch("openai.LLM", return_value=MagicMock()), \
             patch("cartesia.TTS", return_value=MagicMock()), \
             patch("turn_detector.EOUModel", return_value=MagicMock()):

            # Run the entrypoint function
            await entrypoint(mock_job_context)

            # Check if the room was created (by verifying the call to agent.start)
            mock_agent.start.assert_called_once_with(mock_job_context.room, mock_participant)

            # Check if the agent said the greeting message
            mock_agent.say.assert_called_once_with("Hey, how can I help you today?", allow_interruptions=True)
