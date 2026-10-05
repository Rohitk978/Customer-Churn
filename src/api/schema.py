from pydantic import BaseModel,Field

class CustomerInput(BaseModel):
    """Schema for customer data"""


    gender: str = Field(..., description="Customer gender")
    partner: str = Field(..., description="Whether customer has a partner")
    dependents: str = Field(..., description="Whether customer has dependents")

    phoneservice: str = Field(..., description="Whether customer has phone service")
    multiplelines: str = Field(..., description="Whether customer has multiple lines")

    internetservice: str = Field(
        ...,
        description="Type of internet service"
    )

    onlinesecurity: str = Field(
        ...,
        description="Whether customer has online security"
    )

    onlinebackup: str = Field(
        ...,
        description="Whether customer has online backup"
    )

    deviceprotection: str = Field(
        ...,
        description="Whether customer has device protection"
    )

    techsupport: str = Field(
        ...,
        description="Whether customer has technical support"
    )

    streamingtv: str = Field(
        ...,
        description="Whether customer has streaming TV"
    )

    streamingmovies: str = Field(
        ...,
        description="Whether customer has streaming movies"
    )

    contract: str = Field(
        ...,
        description="Customer contract type"
    )

    paperlessbilling: str = Field(
        ...,
        description="Whether customer uses paperless billing"
    )

    paymentmethod: str = Field(
        ...,
        description="Customer payment method"
    )

    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months customer has stayed"
    )

    monthlycharges: float = Field(
        ...,
        ge=0,
        description="Customer monthly charges"
    )

    totalcharges: float = Field(
        ...,
        ge=0,
        description="Customer total charges"
    )