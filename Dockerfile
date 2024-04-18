FROM python:3.11
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./s2smodels.py /code/
COPY ./macros.py /code/
COPY ./utils/ . /code/ 
COPY ./modules/ . /code/ 
COPY ./models/ . /code/ 
COPY ./data/ . /code/
COPY ./prompts/ . /code/ 
COPY ./customs/ . /code/ 
COPY ./main.py /code/ 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]