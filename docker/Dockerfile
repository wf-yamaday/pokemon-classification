FROM python:3.7

ENV LANG C.UTF-8
ENV TZ Asia/Tokyo

WORKDIR /app
ADD . ./

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
