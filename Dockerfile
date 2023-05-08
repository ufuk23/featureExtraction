FROM python:3.8-slim

RUN apt-get update \
&& apt-get install nano \
&& apt-get clean

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./milvus2_create_vectors.py ./

CMD ["python", "./milvus2_create_vectors.py"]
