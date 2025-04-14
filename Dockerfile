
# # Start from the Python 3.10 slim base image
# FROM python:3.10-slim

# # Set the working directory
# WORKDIR /app

# # Copy the required files into the container
# COPY requirements-dev.txt /app
# COPY models/transformer.pkl /app/models/
# COPY app.py /app
# COPY src/ /app/src/

# # Install the required Python packages
# RUN pip install --no-cache-dir -r requirements-dev.txt

# # Command to run the application
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.10.0
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements-dev.txt,target=requirements-dev.txt \
    python -m pip install -r requirements-dev.txt

# Switch to the non-privileged user to run the application.
USER appuser

# # Copy the source code into the container.
# COPY ./container_models/ ./container_models/
# COPY app.py .
# COPY data_models.py .
# COPY requirements-dev.txt .

# Copy the required files into the container
COPY requirements-dev.txt /app
COPY models/transformer.pkl /app/models/
COPY app.py /app
COPY src/ /app/src/

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]