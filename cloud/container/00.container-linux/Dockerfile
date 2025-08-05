FROM python:3.11-alpine
RUN apk add --no-cache bash curl gcc musl-dev linux-headers jq
RUN pip install fastapi uvicorn psutil python-multipart
VOLUME ["/mycode"]
COPY mycode.py/ mycode/mycode.py
CMD ["sh", "-c", "echo 'start server' && sleep infinity"]