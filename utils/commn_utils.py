s3_prefix = "https://feed-images-01.s3.ap-south-1.amazonaws.com/"
cdn_prefix = "https://d19dlu1w9mnmln.cloudfront.net/"

def get_cdn_url(url):
    return url.replace(s3_prefix, cdn_prefix)