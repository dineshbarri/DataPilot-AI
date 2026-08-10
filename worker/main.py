from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from datapilot.safety import UnsafeCodeError, validate_generated_expression

app = FastAPI(title="DataPilot restricted worker", docs_url=None, redoc_url=None)


class CalculationRequest(BaseModel):
    expression: str = Field(max_length=2_000)
    variables: dict[str, float] = Field(default_factory=dict)


@app.get("/health")
def health():
    return {"status": "healthy", "network": "disabled-by-container"}


@app.post("/calculate")
def calculate(request: CalculationRequest):
    try:
        tree = validate_generated_expression(request.expression)
        value = eval(  # noqa: S307 - restricted AST + isolated container + empty builtins
            compile(tree, "<restricted-expression>", "eval"),
            {"__builtins__": {}},
            dict(request.variables),
        )
        if not isinstance(value, (int, float, bool)):
            raise UnsafeCodeError("Expression must return a scalar number or boolean.")
        return {"result": value}
    except (UnsafeCodeError, ArithmeticError, NameError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
