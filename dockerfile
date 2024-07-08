FROM python:3.12

RUN apt-get update && \
    apt-get install -y libhdf5-dev && \
    apt-get clean

WORKDIR /mysalak_flask_api

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["waitress-serve", "--host", "0.0.0.0", "--port", "8888", "server:app"]
# CMD ["uvicorn", "server:asgi_app", "--host", "0.0.0.0", "--port", "8000"]

