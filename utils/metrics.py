import time
import boto3
import os
from functools import wraps
from datetime import datetime
from django.conf import settings

cloudwatch = boto3.client('cloudwatch', region_name='ap-south-1')

class MetricsUtil:
    @staticmethod
    def put_metric(name, value, dimensions, unit='Count'):
        """
        Send a single metric to CloudWatch
        """

        # Skip metrics in local development
        if settings.DEBUG or os.getenv('ENVIRONMENT') == 'local':
            print(f"[LOCAL] Metric - Name: {name}, Value: {value}, Dimensions: {dimensions}")
            return
        
        try:
            cloudwatch.put_metric_data(
                Namespace='Slayd/API',
                MetricData=[{
                    'MetricName': name,
                    'Value': value,
                    'Unit': unit,
                    'Dimensions': dimensions,
                    'Timestamp': datetime.now(datetime.UTC)
                }]
            )
        except Exception as e:
            # Log the error but don't raise it to avoid disrupting the main flow
            print(f"Failed to put CloudWatch metric: {str(e)}")

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