FROM python:3.12

RUN apt-get update && \
    apt-get install -y libhdf5-dev && \
    apt-get clean

WORKDIR /mysalak_flask_api

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CMD ["python", "server.py"]
CMD ["waitress-serve", "--host", "127.0.0.1", "server:app"]

