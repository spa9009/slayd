import time
import logging
import os
from datetime import datetime
from django.conf import settings
from utils.metrics import MetricsUtil

logger = logging.getLogger(__name__)

class MetricsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        logger.info("MetricsMiddleware initialized")

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
        
        # Send request metric
        logger.debug("Sending RequestCount metric")
        MetricsUtil.put_metric('RequestCount', 1, [
            {'Name': 'Endpoint', 'Value': request.path},
            {'Name': 'Method', 'Value': request.method}
        ])
        
        response = self.get_response(request)
        
        duration = (time.time() - start_time) * 1000  # milliseconds
        
        # Send response metrics
        logger.debug("Sending ResponseTime metric")
        MetricsUtil.put_metric('ResponseTime', duration, [
            {'Name': 'Endpoint', 'Value': request.path}
        ], 'Milliseconds')
        
        logger.debug("Sending StatusCode metric")
        MetricsUtil.put_metric(f'StatusCode', 1, [
            {'Name': 'Endpoint', 'Value': request.path},
            {'Name': 'StatusCode', 'Value': str(response.status_code)}
        ])
        
        return response

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR') 