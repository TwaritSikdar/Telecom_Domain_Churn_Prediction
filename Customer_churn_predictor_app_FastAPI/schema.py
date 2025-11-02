# Importing the library
from pydantic import BaseModel

# Creating a class InputModel
class CustomerInfo(BaseModel):
    """
    Defines the expected input schema for the /predict endpoint, covering all
    features used in the model preprocessing pipeline.
    """
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float