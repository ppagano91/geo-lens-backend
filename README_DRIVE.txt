README — Bandas satelitales guardadas en Drive para GeoLens
Fecha: 2026-08-05

Objetivo de esta carpeta
------------------------
Esta carpeta de Drive se usa para guardar archivos comprimidos de escenas satelitales descargadas desde la web, principalmente Landsat 8 y eventualmente Sentinel-2.

Los archivos se guardan comprimidos en Drive porque las bandas GeoTIFF ocupan mucho espacio en el disco local. La idea es conservar una copia original o semioriginal de las escenas descargadas, sin llenar la máquina de trabajo.

Para qué sirven estos archivos
------------------------------
Estos archivos contienen las bandas raster necesarias para probar y alimentar GeoLens, una app GIS/teledetección que permite registrar escenas, detectar bandas, calcular índices espectrales y generar resultados descargables.

Con estas bandas, GeoLens puede calcular índices como:

- NDVI: vegetación
- NDWI: agua / humedad superficial
- NBR: áreas quemadas o severidad de incendio
- NDMI: humedad de vegetación / contenido de agua

En el caso de Landsat 8 Collection 2 Level-2 Surface Reflectance, las bandas más importantes para GeoLens son:

- SR_B2: Blue
- SR_B3: Green
- SR_B4: Red
- SR_B5: NIR
- SR_B6: SWIR1
- SR_B7: SWIR2

GeoLens usa esas bandas para resolver los roles espectrales de cada índice. Por ejemplo:

- NDVI usa NIR y RED    -> Landsat 8: SR_B5 y SR_B4
- NDWI usa GREEN y NIR  -> Landsat 8: SR_B3 y SR_B5
- NBR usa NIR y SWIR2   -> Landsat 8: SR_B5 y SR_B7
- NDMI usa NIR y SWIR1  -> Landsat 8: SR_B5 y SR_B6

Por qué conservar los comprimidos
---------------------------------
Conviene conservar los comprimidos porque:

1. Funcionan como backup de las escenas descargadas.
2. Permiten repetir pruebas en GeoLens sin volver a descargar desde EarthExplorer u otra fuente.
3. Permiten reconstruir el set de bandas si se borran archivos locales.
4. Evitan ocupar espacio permanente en el disco de la máquina.
5. Ayudan a mantener separado el archivo original de los derivados generados por la app.

Qué NO guarda GeoLens en la base de datos
-----------------------------------------
GeoLens no guarda los GeoTIFF completos dentro de PostgreSQL.

La base de datos guarda metadata y referencias, por ejemplo:

- escena
- sensor
- fecha de adquisición
- footprint
- band_key
- band_name
- asset_path
- metadata técnica

Los archivos raster reales viven en el storage de la app. En desarrollo, ese storage es DATA_ROOT.

Ejemplo conceptual:

DB:
  raster_scenes
  raster_bands
  asset_path = sample/scenes/landsat8_lc08_225084/SR_B5.tif

Storage local:
  DATA_ROOT/sample/scenes/landsat8_lc08_225084/SR_B5.tif

Dónde descomprimir para usar con GeoLens
----------------------------------------
En el entorno local actual, GeoLens espera leer escenas desde DATA_ROOT.

Si DATA_ROOT está configurado como:

  DATA_ROOT=../data

Y el backend se ejecuta desde:

  geo-lens-backend/

Entonces la carpeta real suele quedar en:

  GeoLens/data/

Para probar una escena local, descomprimir o copiar las bandas en una carpeta como:

  GeoLens/data/sample/scenes/<nombre_escena>/

Ejemplo:

  GeoLens/data/sample/scenes/landsat8_lc08_225084_20260510/
    LC08_..._SR_B2.TIF
    LC08_..._SR_B3.TIF
    LC08_..._SR_B4.TIF
    LC08_..._SR_B5.TIF
    LC08_..._SR_B6.TIF
    LC08_..._SR_B7.TIF
    LC08_..._MTL.txt

El nombre de carpeta puede ser libre, pero conviene que sea claro y único, por ejemplo:

  landsat8_caba_20260510
  landsat8_bahia_pre_inundacion
  landsat8_bahia_post_inundacion
  landsat8_lc08_225084_20260510

Cómo descomprimir en Windows
----------------------------
Opción 1: desde el Explorador de Windows

1. Descargar el ZIP/TAR desde Drive.
2. Click derecho sobre el archivo.
3. Elegir "Extraer todo...".
4. Seleccionar una carpeta dentro de:

   GeoLens/data/sample/scenes/<nombre_escena>/

Opción 2: desde PowerShell si es .zip

  Expand-Archive -Path "C:\ruta\archivo.zip" -DestinationPath "C:\Proyectos\GIS Apps\GeoLens\data\sample\scenes\landsat8_caba_20260510"

Opción 3: desde Git Bash o WSL si es .tar.gz

  tar -xzf archivo.tar.gz -C "/c/Proyectos/GIS Apps/GeoLens/data/sample/scenes/landsat8_caba_20260510"

Cómo registrar la escena en GeoLens
-----------------------------------
Una vez que las bandas están dentro de DATA_ROOT, usar la pestaña "Ingesta" de GeoLens.

Ejemplo de carga:

  scene_path = sample/scenes/landsat8_caba_20260510
  source     = landsat-8
  name       = Landsat 8 CABA 2026-05-10

Importante:

- scene_path debe ser relativo a DATA_ROOT.
- No usar rutas absolutas.
- No usar "..".
- Para Landsat 8, source debe ser landsat-8.

Al registrar la escena, GeoLens debería:

1. Leer la carpeta.
2. Encontrar los GeoTIFF.
3. Detectar las bandas SR_B2 a SR_B7.
4. Leer metadata del MTL.txt si existe.
5. Validar que las bandas tengan mismo CRS, tamaño y transform.
6. Crear un registro en raster_scenes.
7. Crear registros en raster_bands.
8. Mostrar índices compatibles.

Después de registrar
--------------------
Luego de registrar la escena:

1. Ir a la pestaña "Índices".
2. Seleccionar la escena registrada.
3. Elegir NDVI, NDWI, NBR o NDMI.
4. Ejecutar:

   - Calcular
   - Calcular y guardar
   - Generar preview
   - Ver preview
   - Descargar GeoTIFF o PNG

Los derivados se guardan en:

  DATA_ROOT/derived/scenes/<scene_id>/

Ejemplo:

  GeoLens/data/derived/scenes/<scene_id>/ndvi.tif
  GeoLens/data/derived/scenes/<scene_id>/ndvi.png

Qué archivos conservar en Drive
-------------------------------
Conservar preferentemente:

- ZIP/TAR original descargado.
- Bandas SR_B2 a SR_B7 si fueron guardadas sueltas.
- Archivo MTL.txt.
- Notas del origen: sensor, fecha, path/row, AOI o evento analizado.

Ejemplo de nombre recomendado para comprimidos:

  landsat8_caba_2026-05-10_lc08_225084.zip
  landsat8_bahia_pre_inundacion_YYYY-MM-DD.zip
  landsat8_bahia_post_inundacion_YYYY-MM-DD.zip

Notas importantes
-----------------
- No subir la carpeta data/ completa al repositorio Git.
- Los GeoTIFF pueden ser pesados y deberían quedar fuera del repo.
- DATA_ROOT es storage local/dev, no una carpeta que el usuario final debería manipular en una versión productiva.
- En el futuro, GeoLens debería permitir upload desde la UI o ingesta desde catálogos/STAC, para que el usuario no tenga que copiar archivos manualmente.
- Drive funciona como backup externo y repositorio manual de escenas descargadas.

Resumen rápido
--------------
Estos comprimidos están en Drive porque son escenas satelitales pesadas usadas para probar y alimentar GeoLens.

Para usarlos:

1. Descargar el comprimido desde Drive.
2. Descomprimirlo en GeoLens/data/sample/scenes/<nombre_escena>/.
3. Abrir GeoLens.
4. Ir a Ingesta.
5. Registrar la escena con source=landsat-8.
6. Usar la escena en Índices.
7. Calcular, guardar, visualizar y descargar resultados.
