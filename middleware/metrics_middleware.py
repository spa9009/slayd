import time
import logging
import boto3
from datetime import datetime
from django.conf import settings

logger = logging.getLogger('api.metrics')
cloudwatch = boto3.client('cloudwatch', region_name='ap-south-1')  # Update with your region

class MetricsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        
        # Log request details
        request_data = {
            'path': request.path,
            'method': request.method,
            'query_params': dict(request.GET.items()),
            'ip': self.get_client_ip(request),
            'user_agent': request.META.get('HTTP_USER_AGENT', ''),
        }
        
        logger.info(f"Incoming request: {request_data}")
        
        # Send request metric to CloudWatch
        self.put_metric('RequestCount', 1, request.path)
        
        response = self.get_response(request)
        
        duration = (time.time() - start_time) * 1000  # milliseconds
        
        # Send response metrics to CloudWatch
        self.put_metric('ResponseTime', duration, request.path)
        self.put_metric(f'StatusCode_{response.status_code}', 1, request.path)
        
        return response

    def put_metric(self, name, value, endpoint):
        try:
            cloudwatch.put_metric_data(
                Namespace='Slayd/API',
                MetricData=[{
                    'MetricName': name,
                    'Value': value,
                    'Unit': 'Milliseconds' if name == 'ResponseTime' else 'Count',
                    'Dimensions': [
                        {
                            'Name': 'Endpoint',
                            'Value': endpoint
                        },
                        {
                            'Name': 'Environment',
                            'Value': settings.ENVIRONMENT  # 'production', 'staging', etc.
                        }
                    ],
                    'Timestamp': datetime.utcnow()
                }]
            )
        except Exception as e:
            logger.error(f"Failed to put CloudWatch metric: {str(e)}")

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR') 