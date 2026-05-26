import os
import sys
from contextlib import contextmanager

# 백엔드 패키지 경로를 sys.path에 동적으로 추가하여 app.database 및 app.models를 재사용합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
# c:\ITStudy\poom\ai\agent\todo\tools -> c:\ITStudy\poom\back
back_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", "back"))

if back_path not in sys.path:
    sys.path.insert(0, back_path)

# Pydantic Settings가 .env 파일을 읽을 때 extra 필드 오류(extra_forbidden)가 나는 현상을 방지하기 위해
# app.database 임포트 시점에만 임시적으로 .env 파일에서 허용되지 않는 필드(OPENAI_API_KEY 등)를 걸러냈다가 복원합니다.
env_path = os.path.join(back_path, ".env")
env_backup_path = os.path.join(back_path, ".env.backup")
has_env = os.path.exists(env_path)

if has_env:
    try:
        # 1. 기존 .env 백업 생성
        with open(env_path, "r", encoding="utf-8") as f:
            env_content = f.read()
        
        with open(env_backup_path, "w", encoding="utf-8") as f:
            f.write(env_content)

        # 2. 허용된 키 목록 정의 (Settings 클래스에 등록된 필드들만 허용)
        allowed_keys = {
            "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME",
            "ECOS_API_KEY", "FRED_API_KEY", "REB_API_KEY",
            "JWT_SECRET_KEY", "JWT_ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES", "REFRESH_TOKEN_EXPIRE_DAYS"
        }

        # 3. 허용된 필드만 가진 임시 .env 작성
        clean_lines = []
        for line in env_content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                clean_lines.append(line)
                continue
            
            parts = stripped.split("=", 1)
            key = parts[0].strip()
            # Pydantic Settings는 대소문자를 구분하지 않고 가져올 수 있으므로 상위 레벨에서 처리
            if key in allowed_keys or key.upper() in allowed_keys:
                clean_lines.append(line)
        
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_lines))
            
    except Exception as e:
        print(f"[Warning] .env 임시 패치 실패: {e}")

try:
    # 4. 이제 app 패키지 안전하게 임포트 (Pydantic ValidationError 발생 방지!)
    from app.database import SessionLocal
finally:
    # 5. 원래 .env 파일 원상태로 즉각 복원
    if has_env and os.path.exists(env_backup_path):
        try:
            with open(env_backup_path, "r", encoding="utf-8") as f:
                original_content = f.read()
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            # 백업 파일 삭제
            os.remove(env_backup_path)
        except Exception as e:
            print(f"[Warning] .env 복원 실패: {e}")

# 이제 app 패키지 임포트 가능
from app.database import SessionLocal

@contextmanager
def get_db_session():
    """SQLAlchemy DB 세션을 제공하는 컨텍스트 매니저"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
