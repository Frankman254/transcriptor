"""
Servidor FastAPI del transcriptor local: montaje de la app y las tres
familias de rutas.  El trabajo real está en `pipelines`, `routes_*` y los
módulos de motor; aquí solo se cablea.
"""
import config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import routes_jobs
import routes_serve
import routes_system

app = FastAPI(title="Transcriptor local")

# Transcriptor es el motor; Lyrixa y el renderer son clientes que corren en
# sus propios puertos, así que sin CORS no pueden llamarlo desde el navegador.
# El permiso es solo para localhost (ver `config`): el servidor escucha en
# loopback, de modo que la única forma de que una página remota llegase aquí
# sería a través del navegador del usuario, y ese es justo el caso que se
# niega. Sin credenciales: esta API no usa cookies ni sesiones.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_origin_regex=config.ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


app.include_router(routes_system.router)
app.include_router(routes_jobs.router)
app.include_router(routes_serve.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
