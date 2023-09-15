FROM python:3.11.5

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 3000


CMD ["python", "AcharyaMeasuringPrototype.py"]
