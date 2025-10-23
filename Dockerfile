# Use the latest Python 3.12 slim image
FROM python:3.12-slim

# Add contrib and non-free repos so we can install MS core fonts
RUN echo "deb http://deb.debian.org/debian bullseye main contrib non-free" > /etc/apt/sources.list.d/contrib-nonfree.list

# Install system dependencies: ICU, fontconfig, MS core fonts, and netcat
RUN apt-get update && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get install -y libicu-dev fontconfig ttf-mscorefonts-installer netcat && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Rebuild font cache so fonts are recognized
RUN fc-cache -f -v

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose default ports for FastAPI
EXPOSE 8000 



ENV APP4="ai_interview.main:app"
ENV APP4_PORT=8000


# Command to run all six FastAPI applications concurrently
CMD ["sh", "-c", "uvicorn $APP1 --host 0.0.0.0 --port $APP1_PORT & \
                 wait"]