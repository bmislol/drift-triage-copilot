from pydantic import BaseModel, Field


class BankPredictionRequest(BaseModel):
    # Numeric features (19 features — duration is excluded: recorded post-call, leaks the label)
    age: int
    campaign: int
    # pdays == 999 means "never contacted before" (sentinel, not a real day count).
    # The ML pipeline converts this to a binary never_contacted flag internally.
    pdays: int = Field(..., description="Days since last contact. 999 = never contacted.")
    previous: int
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")

    # Categorical features
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    poutcome: str

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "age": 41,
                "job": "blue-collar",
                "marital": "divorced",
                "education": "basic.4y",
                "default": "unknown",
                "housing": "yes",
                "loan": "no",
                "contact": "telephone",
                "month": "may",
                "day_of_week": "mon",
                "campaign": 1,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp.var.rate": 1.1,
                "cons.price.idx": 93.994,
                "cons.conf.idx": -36.4,
                "euribor3m": 4.857,
                "nr.employed": 5191.0,
            }
        },
    }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
