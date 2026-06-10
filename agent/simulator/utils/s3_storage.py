# utils/s3_storage.py
import boto3
import json
import os
from botocore.exceptions import ClientError

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
s3 = boto3.client("s3")

def s3_read_text(key: str) -> str | None:
    try:
        res = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return res["Body"].read().decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise  # 다른 에러는 그냥 올려보냄

def s3_write_text(key: str, content: str):
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=content.encode("utf-8"))

def s3_read_json(key: str) -> list | None:
    text = s3_read_text(key)
    if text is None:
        return None
    return json.loads(text)

def s3_write_json(key: str, data):
    s3_write_text(key, json.dumps(data, ensure_ascii=False, indent=2))

def s3_delete(key: str):
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise
    except Exception:
        pass

def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return False
        raise   
    except Exception:
        return False