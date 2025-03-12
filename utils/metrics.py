import time
import boto3
import os
from functools import wraps
from datetime import datetime, timezone
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class MetricsUtil:
    _cloudwatch_client = None

    @classmethod
    def get_cloudwatch_client(cls):
        if cls._cloudwatch_client is None:
            try:
                session = boto3.Session()
                cls._cloudwatch_client = session.client('cloudwatch', region_name='ap-south-1')
                logger.info("Successfully created CloudWatch client")
            except Exception as e:
                logger.error(f"Failed to create CloudWatch client: {str(e)}", exc_info=True)
        return cls._cloudwatch_client

    @staticmethod
    def put_metric(name, value, dimensions, unit='Count'):
        """
        Send a single metric to CloudWatch
        """
        if settings.DEBUG:
            logger.info(f"[LOCAL] Metric - Name: {name}, Value: {value}, Dimensions: {dimensions}")
            return

        try:
            client = MetricsUtil.get_cloudwatch_client()
            if client:
                metric_data = {
                    'Namespace': 'Slayd/API',
                    'MetricData': [{
                        'MetricName': name,
                        'Value': value,
                        'Unit': unit,
                        'Dimensions': dimensions,
                        'Timestamp': datetime.now(timezone.utc)
                    }]
                }
                client.put_metric_data(**metric_data)
            else:
                logger.error("CloudWatch client is None")
        except Exception as e:
            logger.error(f"Failed to put CloudWatch metric: {str(e)}", exc_info=True)

    @staticmethod
    def track_execution_time(endpoint_name):
        """
        Decorator to track execution time of a function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = (time.time() - start_time) * 1000
                    MetricsUtil.put_metric(
                        'ProcessingTime',
                        duration,
                        [{'Name': 'Endpoint', 'Value': endpoint_name}],
                        'Milliseconds'
                    )
            return wrapper
        return decorator

    @staticmethod
    def record_success(endpoint_name, additional_dimensions=None):
        """
        Record a successful operation
        """
        dimensions = [{'Name': 'Endpoint', 'Value': endpoint_name}]
        if additional_dimensions:
            dimensions.extend(additional_dimensions)
        MetricsUtil.put_metric('Success', 1, dimensions)

    @staticmethod
    def record_failure(endpoint_name, error_type, additional_dimensions=None):
        """
        Record a failed operation
        """
        dimensions = [
            {'Name': 'Endpoint', 'Value': endpoint_name},
            {'Name': 'ErrorType', 'Value': error_type}
        ]
        if additional_dimensions:
            dimensions.extend(additional_dimensions)
        MetricsUtil.put_metric('Failure', 1, dimensions) 