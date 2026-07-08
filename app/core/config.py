from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "GeoChange Analyzer API"
    version: str = "0.1.0"
    app_env: str = "local"
    database_url: str = (
        "postgresql+psycopg://geochange:geochange@localhost:5432/geochange"
    )


settings = Settings()
