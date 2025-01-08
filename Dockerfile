FROM python:3.9

WORKDIR /app

ADD . /app
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD [ "python","app.py" ]
