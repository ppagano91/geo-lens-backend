from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "GeoChange Analyzer API"
    version: str = "0.1.0"
    app_env: str = "local"
    database_url: str = (
        "postgresql+psycopg://geochange:geochange@localhost:5432/geochange"
    )
    cors_origins: str = "http://localhost:5173"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def normalize_cors_origins(cls, value: object) -> str:
        if isinstance(value, list):
            return ",".join(str(origin).strip() for origin in value if str(origin).strip())
        if value is None:
            return "http://localhost:5173"
        return str(value)

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
