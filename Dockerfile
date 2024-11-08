FROM python:3

RUN pip install requests
RUN pip install pydantic

RUN apt-get update && apt-get install -y \
    libenchant-2-2  # Enchant library for pyenchant

RUN pip install pyenchant

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]