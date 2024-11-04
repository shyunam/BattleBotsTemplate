FROM python:3.8.12

RUN pip install requests
RUN pip install pydantic
# Install Rust and Cargo (required by some transformers components)
# Install curl to fetch the Rust installer
#RUN apt-get update && apt-get install -y curl libsentencepiece-dev libssl-dev && \
    #curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    #export PATH="/root/.cargo/bin:${PATH}" && \
    #pip install 'transformers[torch]'
RUN pip install 'transformers[torch]'
RUN pip install emoji==0.6.0

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]