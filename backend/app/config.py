from pydantic import BaseSettings

# configure env vars to match a schema
class Settings(BaseSettings):
    db_host: str
    db_port: str
    db_name: str
    db_username: str
    db_password: str
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    admin_id: int

    class Config:
        env_file = ".env"

settings = Settings()
