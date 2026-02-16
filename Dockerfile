FROM langflowai/langflow:latest

COPY components/ /app/custom_components/

RUN mkdir -p /app/langflow/guidelines && chmod 777 /app/langflow/guidelines

ENV LANGFLOW_COMPONENTS_PATH=/app/custom_components
ENV LANGFLOW_AUTO_LOGIN=true
ENV PYTHONPATH=/app/custom_components/copywriting

EXPOSE 7860

CMD ["python", "-m", "langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
