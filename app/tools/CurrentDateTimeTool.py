from datetime import datetime, timezone
from crewai.tools import BaseTool
from pydantic.v1 import BaseModel

class _CurrentDateTimeArgs(BaseModel):
    pass

class CurrentDateTimeTool(BaseTool):
    name: str = "CurrentDateTime"
    description: str = (
        "Returns the current UTC and local date/time so the agent is aware of 'now'. "
        "Use this before asking about recent events or when crafting time-sensitive search queries."
    )
    # Pydantic v2 requires preserving type annotation when overriding BaseTool field definitions
    args_schema: type[_CurrentDateTimeArgs] = _CurrentDateTimeArgs

    def _run(self) -> str:
        now_utc = datetime.now(timezone.utc)
        local_now = datetime.now()
        payload = {
            "utc_iso": now_utc.isoformat(),
            "local_iso": local_now.isoformat(),
            "date": local_now.strftime("%Y-%m-%d"),
            "time": local_now.strftime("%H:%M:%S"),
            "weekday": local_now.strftime("%A"),
            "timestamp": int(now_utc.timestamp())
        }
        # Return as a compact string for LLM consumption
        return (
            f"Current Date/Time Context => UTC: {payload['utc_iso']}; Local: {payload['local_iso']}; "
            f"Date: {payload['date']} ({payload['weekday']})"
        )

    async def _arun(self):  # pragma: no cover - async not required now
        return self._run()
