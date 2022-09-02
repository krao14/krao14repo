# dockerdile, Image, Container
FROM --platform=linux/amd64 python:3.9

WORKDIR /code

EXPOSE 8501

ADD app1.py .
ADD framingham_heart_disease.csv .

COPY . /code

RUN pip install --upgrade pip \
    && pip install numpy \
    && pip install pandas \
    && pip install streamlit==1.9.2 \
    && pip install sklearn \
    && pip install conda \
    && pip install watchdog

ENTRYPOINT ["streamlit", "run"]
CMD ["app1.py"]

