from pydantic import BaseModel, Field
from typing import Optional


class Preference(BaseModel):
    title: Optional[str] = Field(
        None,
        description="Short descriptive title of the preference, e.g. 'Bollywood Movies'"
    )
    content: Optional[str] = Field(
        None,
        description="Detailed description of the user's preference"
    )
    query_detected: bool = Field(
        default=False,
        description="True if the user also asked a question in the same message"
    )
