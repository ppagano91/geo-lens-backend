from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "GeoLens Analyzer API"
    version: str = "0.1.0"
    app_env: str = "local"
    database_url: str = (
        "postgresql+psycopg://geochange:geochange@localhost:5432/geochange"
    )
    cors_origins: str = "http://localhost:5173"
    # Relative paths in raster_bands.asset_path resolve against this root
    # via AssetStorageService (local filesystem storage for now).
    data_root: str = "../data"
    # Max size per uploaded file for POST /ingest/upload-scene (0 = unlimited).
    max_upload_file_bytes: int = 536_870_912  # 512 MiB

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

    @property
    def data_root_path(self) -> Path:
        """Absolute, resolved DATA_ROOT (relative values resolve from process cwd)."""
        return Path(self.data_root).expanduser().resolve()


settings = Settings()
