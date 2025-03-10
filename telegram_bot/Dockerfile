# Use the official Ubuntu image from the Docker Hub
FROM ubuntu:24.04

# Install system packages
RUN apt-get update \
    && apt-get install -y python3-pip libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install required Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Copy the current directory contents into the container at /app
COPY . .

# Command to run the Telegram application
CMD ["python3", "bot.py"]