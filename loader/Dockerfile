FROM python:3.8
ENV PLATFORM "loader"
RUN apt-get update && apt-get install -y

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
RUN pip3 install -e ./api/
ENTRYPOINT python3 start.py