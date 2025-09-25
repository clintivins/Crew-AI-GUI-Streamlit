from pydantic import BaseModel, Field, model_validator
from crewai_tools import RagTool
from typing import Any, Optional, Type
from crewai_tools.tools.rag.rag_tool import Adapter
from enum import Enum

"""CSVSearchToolEnhanced

This module depends on the optional 'embedchain' package. The original implementation
performed hard imports which caused the entire Streamlit app to fail on startup when
`embedchain` was not installed (common on Python 3.13 where older embedchain versions
try to compile an older tiktoken requiring a Rust toolchain).

We now gracefully degrade: if embedchain is unavailable the tool remains importable
but any attempt to use it will return an informative message instead of crashing
the whole application.
"""

EMBEDCHAIN_AVAILABLE = True
try:  # Attempt to import embedchain lazily / defensively
    from embedchain.models.data_type import DataType  # type: ignore
    from embedchain import App  # type: ignore
except Exception:  # Broad except to cover ImportError plus potential version errors
    EMBEDCHAIN_AVAILABLE = False

    class _PlaceholderDataType(Enum):  # Minimal stand‑in so attribute access is safe
        CSV = "csv"

    DataType = _PlaceholderDataType  # type: ignore
    App = None  # type: ignore


class CSVEmbedchainAdapter(Adapter):
    """Safe adapter wrapper. If embedchain isn't available we short‑circuit.
    """

    embedchain_app: Any  # Using Any because App may be a placeholder (None) when unavailable
    summarize: bool = False
    src: Optional[str] = None

    def _unavailable(self) -> str:
        return (
            "The 'embedchain' dependency is not installed (or failed to load). "
            "Install it manually with 'pip install embedchain' (Python 3.11/3.12 recommended) "
            "to enable CSV semantic search."
        )

    def query(self, question: str) -> str:
        if not EMBEDCHAIN_AVAILABLE or self.embedchain_app is None:
            return self._unavailable()
        where = (
            {"app_id": self.embedchain_app.config.id, "url": self.src}
            if self.src
            else None
        )
        result, sources = self.embedchain_app.query(
            question, citations=True, dry_run=(not self.summarize), where=where
        )
        if self.summarize:
            return result
        return "\n\n".join([source[0] for source in sources])

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not EMBEDCHAIN_AVAILABLE or self.embedchain_app is None:
            return  # Silently ignore when unavailable
        self.src = args[0] if args else None
        self.embedchain_app.add(*args, **kwargs)

class FixedCSVSearchToolSchema(BaseModel):
    """Input for CSVSearchTool."""

    query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the CSV's content",
    )

class CSVSearchToolSchema(FixedCSVSearchToolSchema):
    """Input for CSVSearchTool."""

    csv: str = Field(..., description="Mandatory csv path you want to search")

class CSVSearchToolEnhanced(RagTool):
    name: str = "Search a CSV's content"
    description: str = (
        "A tool that can be used to semantic search a query from a CSV's content."
    )
    args_schema: Type[BaseModel] = CSVSearchToolSchema

    @model_validator(mode="after")
    def _set_default_adapter(self):
        if isinstance(self.adapter, RagTool._AdapterPlaceholder):
            app_instance = None
            if EMBEDCHAIN_AVAILABLE and App is not None:
                try:
                    app_instance = App.from_config(config=self.config) if self.config else App()  # type: ignore
                except Exception:
                    # Fallback to unavailable state
                    app_instance = None
            self.adapter = CSVEmbedchainAdapter(
                embedchain_app=app_instance, summarize=self.summarize
            )
        return self

    def __init__(self, csv: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None, **kwargs):
        if csv and description is None:
            kwargs["description"] = f"A tool that can be used to semantic search a query the {csv} CSV's content."
        if name:
            kwargs["name"] = name
        if description:
            kwargs["description"] = description
        if csv:
            kwargs["data_type"] = getattr(DataType, "CSV", "csv")
            kwargs["args_schema"] = FixedCSVSearchToolSchema
            super().__init__(**kwargs)
            try:
                self.add(csv)
            except Exception:
                pass
        else:
            super().__init__(**kwargs)

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().add(*args, **kwargs)

    def _before_run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        if "csv" in kwargs:
            self.add(kwargs["csv"])

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        if not "query" in kwargs:
            return "Please provide a query to search the CSV's content."
        if not "csv" in kwargs and not self.args_schema == FixedCSVSearchToolSchema:
            return "Please provide a CSV to search."
        if not EMBEDCHAIN_AVAILABLE:
            return (
                "CSV semantic search tool is disabled: missing 'embedchain'. "
                "Install with 'pip install embedchain' (Python 3.11/3.12) to enable."
            )
        return super()._run(**kwargs)
    

    

