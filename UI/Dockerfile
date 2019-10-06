FROM python:3

WORKDIR /usr/src/app/

COPY requirements.txt /usr/src/app/

RUN pip install -r /usr/src/app/requirements.txt

COPY app.py /usr/src/app/

EXPOSE 3000

CMD [ "python", "/usr/src/app/app.py" ]