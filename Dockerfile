FROM python:3.13

WORKDIR /opt/car_price_api

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt  ./requirements.txt

RUN pip install  --no-cache-dir  -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /opt/car_price_api

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
