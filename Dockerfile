
FROM python:3.13

WORKDIR /opt/car_price_api

RUN useradd --create-home --shell /bin/bash appuser

ARG MODEL_PACKAGE_VERSION=0.1.0

COPY requirements.txt ./requirements.txt

RUN sed -i '/^car-price-model==/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "car-price-model==${MODEL_PACKAGE_VERSION}"

COPY . .

RUN chown -R appuser:appuser /opt/car_price_api

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
