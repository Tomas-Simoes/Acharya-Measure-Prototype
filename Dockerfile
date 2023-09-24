FROM public.ecr.aws/lambda/python:3.11.2023.09.12.11

WORKDIR /app

COPY Models/best.pt /app
COPY AcharyaMeasuringPrototype.py /app
COPY Utlis.py /app


RUN pip install --upgrade pip

RUN pip cache purge
RUN pip install 'ultralytics[yolo]~=8.0.147'
RUN pip install -r requirements.txt

EXPOSE 3000

CMD [ "AcharyaMeasuringPrototype.init" ]