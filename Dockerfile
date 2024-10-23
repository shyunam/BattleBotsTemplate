FROM python:3.8

RUN pip install requests
RUN pip install pydantic
# Install Rust and Cargo (required by some transformers components)
# Install curl to fetch the Rust installer
RUN apt-get update && apt-get install -y curl

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Add Cargo to PATH
ENV PATH="/root/.cargo/bin:${PATH}"
RUN apt-get install -y libsentencepiece-dev

RUN pip install 'transformers[torch]'

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]