# Use the recommended Python version
FROM python:3.9

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variable for user PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the app files
COPY --chown=user . /app

# Expose the port (not strictly needed since Spaces sets it dynamically)
EXPOSE 7860

# Start FastAPI with Uvicorn using the dynamic port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
