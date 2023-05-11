FROM python:3.8-slim

RUN apt-get update \
&& apt-get install nano \
&& apt-get clean

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./qdrant_store_vectors_to_db.py ./

CMD ["python", "-u", "./qdrant_store_vectors_to_db.py"]
