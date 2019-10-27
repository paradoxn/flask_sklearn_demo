#登陆限制的装饰器
from functools import wraps
from flask import session,redirect,url_for
def upload_file(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        if session.get('file'):
            return func(*args,**kwargs)
        else:
            return '请上传文件'

    return wrapper
