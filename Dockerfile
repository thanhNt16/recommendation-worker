FROM python:3.8.3

ADD requirements.txt /
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app
EXPOSE 5000
CMD [ "python3", "-u", "app.py" ]

