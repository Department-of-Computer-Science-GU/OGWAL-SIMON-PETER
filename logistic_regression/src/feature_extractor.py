import urllib.parse
import re

def extract_features(url):
    features = {}
    
    # Prepend http:// if no scheme is present to parse correctly
    if not url.startswith('http://') and not url.startswith('https://'):
        parsed_url = urllib.parse.urlparse('http://' + url)
    else:
        parsed_url = urllib.parse.urlparse(url)
        
    domain = parsed_url.netloc
    path = parsed_url.path
    
    # urlLen
    features['urlLen'] = len(url)
    
    # isIp
    # Simple regex for IPv4
    is_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(':')[0]) else 0
    features['isIp'] = is_ip
    
    # is@
    features['is@'] = 1 if '@' in url else 0
    
    # isredirect
    # naive check: is there another http/https in the URL? Or '//' in path?
    features['isredirect'] = 1 if '//' in path else 0
    
    # haveDash
    features['haveDash'] = 1 if '-' in domain else 0
    
    # domainLen
    features['domainLen'] = len(domain)
    
    # nosOfSubdomain
    # Count of dots in domain
    features['nosOfSubdomain'] = domain.count('.')
    
    return features
