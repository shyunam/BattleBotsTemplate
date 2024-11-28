FROM python:3.8.12

RUN pip install requests
RUN pip install pydantic

RUN pip install transformers[torch]

RUN apt-get update && apt-get install -y \
    libenchant-2-2

RUN pip install pyenchant

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]